# function to calculate the c-index
def c_statistic_harrell(pred, labels):
    total = 0
    matches = 0

    # count the matches in the ordering
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[j] > 0 and abs(labels[i]) > labels[j]:
                total += 1
                if pred[j] > pred[i]:
                    matches += 1
    return matches / total



# compute mask that represents each sample's risk set
def _make_riskset(time: np.ndarray) -> np.ndarray:
    assert time.ndim == 1, "expected 1D array"
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set


# callable input function that computes the risk set for each batch
class InputFunction:

    def __init__(self,
                 images: np.ndarray,
                 time: np.ndarray,
                 event: np.ndarray,
                 batch_size: int = 32,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 seed: int = 89) -> None:
        self.images = images
        self.time = time
        self.event = event
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

    def size(self) -> int:
        return self.images.shape[0]

    def steps_per_epoch(self) -> int:
        return int(np.floor(self.size() / self.batch_size))

    # compute risk set for samples in batch
    def _get_data_batch(self, index: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        time = self.time[index]
        event = self.event[index]
        images = self.images[index]
        labels = {
            "label_event": event.astype(np.int32),
            "label_time": time.astype(np.float32),
            "label_riskset": _make_riskset(time)
        }
        return images, labels

    # generator that yields one batch at a time
    def _iter_data(self) -> Iterable[Tuple[np.ndarray, Dict[str, np.ndarray]]]:

        index = np.arange(self.size())
        rnd = np.random.RandomState(self.seed)

        if self.shuffle:
            rnd.shuffle(index)
        for b in range(self.steps_per_epoch()):
            start = b * self.batch_size
            idx = index[start:(start + self.batch_size)]
            yield self._get_data_batch(idx)

        if not self.drop_last:
            start = self.steps_per_epoch() * self.batch_size
            idx = index[start:]
            yield self._get_data_batch(idx)

    # return shapes of data returned by `self._iter_data`
    def _get_shapes(self) -> Tuple[tf.TensorShape, Dict[str, tf.TensorShape]]:

        batch_size = self.batch_size if self.drop_last else None
        d = input_shape
        images = tf.TensorShape([batch_size, d])

        labels = {k: tf.TensorShape((batch_size,))
                  for k in ("label_event", "label_time")}
        labels["label_riskset"] = tf.TensorShape((batch_size, batch_size))
        return images, labels

    # return dtypes of data returned by `self._iter_data`
    def _get_dtypes(self) -> Tuple[tf.DType, Dict[str, tf.DType]]:

        labels = {"label_event": tf.int32,
                  "label_time": tf.float32,
                  "label_riskset": tf.bool}
        return tf.float32, labels

    # create dataset from generator
    def _make_dataset(self) -> tf.data.Dataset:

        ds = tf.data.Dataset.from_generator(
            self._iter_data,
            self._get_dtypes(),
            self._get_shapes()
        )
        return ds

    def __call__(self) -> tf.data.Dataset:
        return self._make_dataset()


# normalize risk scores to avoid exp underflowing
def safe_normalize(x: tf.Tensor) -> tf.Tensor:
    x_min = tf.reduce_min(x, axis=0)
    c = tf.zeros_like(x_min)
    norm = tf.where(x_min < 0, -x_min, c)
    return x + norm


# compute logsumexp across `axis` for entries where `mask` is true
def logsumexp_masked(risk_scores: tf.Tensor,
                     mask: tf.Tensor,
                     axis: int = 0,
                     keepdims: Optional[bool] = None) -> tf.Tensor:
    risk_scores.shape.assert_same_rank(mask.shape)

    with tf.name_scope("logsumexp_masked"):
        mask_f = tf.cast(mask, risk_scores.dtype)
        risk_scores_masked = tf.math.multiply(risk_scores, mask_f)
        # for numerical stability, substract the maximum value before taking the exponential
        amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
        risk_scores_shift = risk_scores_masked - amax
        exp_masked = tf.math.multiply(tf.exp(risk_scores_shift), mask_f)
        exp_sum = tf.reduce_sum(exp_masked, axis=axis, keepdims=True)
        output = amax + tf.math.log(exp_sum)
        if not keepdims:
            output = tf.squeeze(output, axis=axis)
    return output


# negative partial log-likelihood of Cox's proportional hazards model
class CoxPHLoss(tf.keras.losses.Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self,
             y_true: Sequence[tf.Tensor],
             y_pred: tf.Tensor) -> tf.Tensor:

        event, riskset = y_true
        predictions = y_pred

        pred_shape = predictions.shape
        if pred_shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "be 2." % pred_shape.ndims)

        if pred_shape[1] is None:
            raise ValueError("Last dimension of predictions must be known.")

        if pred_shape[1] != 1:
            raise ValueError("Dimension mismatch: Last dimension of predictions "
                             "(received %s) must be 1." % pred_shape[1])

        if event.shape.ndims != pred_shape.ndims:
            raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "equal rank of event (received %s)" % (
                                 pred_shape.ndims, event.shape.ndims))

        if riskset.shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of riskset (received %s) should "
                             "be 2." % riskset.shape.ndims)

        event = tf.cast(event, predictions.dtype)
        predictions = safe_normalize(predictions)

        with tf.name_scope("assertions"):
            assertions = (
                tf.debugging.assert_less_equal(event, 1.),
                tf.debugging.assert_greater_equal(event, 0.),
                tf.debugging.assert_type(riskset, tf.bool)
            )

        # move batch dimension to the end so predictions get broadcast row-wise when multiplying by riskset
        pred_t = tf.transpose(predictions)
        # compute log of sum over risk set for each row
        rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)
        assert rr.shape.as_list() == predictions.shape.as_list()
        losses = tf.math.multiply(event, rr - predictions)

        return losses


# compute concordance index across one epoch
class CindexMetric:

    def reset_states(self) -> None:
        self._data = {
            "label_time": [],
            "label_event": [],
            "prediction": []
        }

    # collect observed time, event indicator and predictions for a batch
    def update_state(self, y_true: Dict[str, tf.Tensor], y_pred: tf.Tensor) -> None:

        self._data["label_time"].append(y_true["label_time"].numpy())
        self._data["label_event"].append(y_true["label_event"].numpy())
        self._data["prediction"].append(tf.squeeze(y_pred).numpy())

    # compute the concordance index across collected values
    def result(self) -> Dict[str, float]:

        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)

        results = concordance_index_censored(
            data["label_event"] == 1,
            data["label_time"],
            data["prediction"])

        result_data = {}
        names = ("cindex", "concordant", "discordant", "tied_risk")
        for k, v in zip(names, results):
            result_data[k] = v

        return result_data


# train and the model similarly to model.fit
class TrainAndEvaluateModel:

    def __init__(self, model, model_dir, train_dataset, eval_dataset,
                 learning_rate, num_epochs):
        self.num_epochs = num_epochs
        self.model_dir = model_dir
        self.model = model
        self.train_ds = train_dataset
        self.val_ds = eval_dataset
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = CoxPHLoss()
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        self.val_cindex_metric = CindexMetric()

    @tf.function
    def train_one_step(self, x, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)

        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            train_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=logits)

        with tf.name_scope("gradients"):
            grads = tape.gradient(train_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return train_loss, logits

    def train_and_evaluate(self):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            optimizer=self.optimizer,
            model=self.model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, str(self.model_dir), max_to_keep=2)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f"Latest checkpoint restored from {ckpt_manager.latest_checkpoint}.")

        train_summary_writer = summary.create_file_writer(
            str(self.model_dir / "train"))
        val_summary_writer = summary.create_file_writer(
            str(self.model_dir / "valid"))

        for epoch in range(self.num_epochs):
            with train_summary_writer.as_default():
                self.train_one_epoch(ckpt.step)

            # Run a validation loop at the end of each epoch.
            with val_summary_writer.as_default():
                self.evaluate(ckpt.step)

        save_path = ckpt_manager.save()
        print(f"Saved checkpoint for step {ckpt.step.numpy()}: {save_path}")

    def train_one_epoch(self, step_counter):
        for x, y in self.train_ds:
            train_loss, logits = self.train_one_step(
                x, y["label_event"], y["label_riskset"])

            step = int(step_counter)
            if step == 0:
                func = self.train_one_step.get_concrete_function(
                    x, y["label_event"], y["label_riskset"])

            # Update training metric.
            self.train_loss_metric.update_state(train_loss)

            # Log every 20 batches.
            if step % 20 == 0:
                mean_loss = self.train_loss_metric.result()
                print(f"step {step}: mean loss = {mean_loss:.4f}")
                summary.scalar("loss", mean_loss, step=step_counter)
                self.train_loss_metric.reset_states()

            step_counter.assign_add(1)

    @tf.function
    def evaluate_one_step(self, x, y_event, y_riskset):
        y_event = tf.expand_dims(y_event, axis=1)
        val_logits = self.model(x, training=False)
        val_loss = self.loss_fn(y_true=[y_event, y_riskset], y_pred=val_logits)
        return val_loss, val_logits

    def evaluate(self, step_counter):
        self.val_cindex_metric.reset_states()

        for x_val, y_val in self.val_ds:
            val_loss, val_logits = self.evaluate_one_step(
                x_val, y_val["label_event"], y_val["label_riskset"])

            # Update val metrics
            self.val_loss_metric.update_state(val_loss)
            self.val_cindex_metric.update_state(y_val, val_logits)

        val_loss = self.val_loss_metric.result()
        summary.scalar("loss",
                       val_loss,
                       step=step_counter)
        self.val_loss_metric.reset_states()

        val_cindex = self.val_cindex_metric.result()
        for key, value in val_cindex.items():
            summary.scalar(key, value, step=step_counter)

        print(f"Validation: loss = {val_loss:.4f}, cindex = {val_cindex['cindex']:.4f}")



# function to do the pretraining of one layer
# autoencoder_cur is the model of the autoencoder to which we add to add and train a layer, current is the dimension of the new bottleneck layer
def pretrain_one_layer(autoencoder_cur, input_shape, current, x_train, x_test, epochs):

    # learning parameters
    loss = 'mean_absolute_error'
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False, name="SGD")

    # input
    inputs = Input(shape=(input_shape,))
    cur_dim = input_shape // 2
    intermediate = 0
    enc_intermediate = inputs

    # encoder
    while cur_dim > current:
        enc_intermediate = Dense(cur_dim, activation='sigmoid')(enc_intermediate)
        enc_intermediate = keras.layers.BatchNormalization()(enc_intermediate)
        cur_dim = cur_dim // 2
        intermediate = intermediate + 1

    # bottleneck
    latent = Dense(current, activation='sigmoid')(enc_intermediate)
    latent = keras.layers.BatchNormalization()(latent)
    encoder_next = Model(inputs, latent)
    dec_intermediate = latent
    cur_dim = cur_dim * 2

    # decoder
    while cur_dim < input_shape:
        dec_intermediate = Dense(cur_dim, activation='sigmoid')(dec_intermediate)
        dec_intermediate = keras.layers.BatchNormalization()(dec_intermediate)
        cur_dim = cur_dim * 2

    # output
    outputs = Dense(input_shape)(dec_intermediate)
    autoencoder_next = Model(inputs, outputs)

    # fix weights to the ones of the previous autoencoder
    if intermediate != 0:
        for i in range(intermediate * 2):
            autoencoder_next.layers[i + 1].set_weights(autoencoder_cur.layers[i + 1].get_weights())

        for i in range(intermediate * 2 - 1):
            autoencoder_next.layers[-i].set_weights(autoencoder_cur.layers[-i].get_weights())

    # show the architecture
    autoencoder_next.summary()

    # compile the model
    autoencoder_next.compile(optimizer=optimizer, loss=loss)

    # train the model
    autoencoder_next.fit(x_train, x_train, epochs=epochs, batch_size=32, shuffle=True, validation_data=(x_test, x_test))

    # return both the new autoencoder and only the new encoder
    return autoencoder_next, encoder_next



# function to do the whole pretraining layerwise
def pretraining(x_train, x_test, input_shape, latent_dim, epochs):

  # initialize the variables for the pretraining of one layer
  current=input_shape//2
  autoencoder_cur=None

  # cycle to pretrain one layer at a time
  while current>=latent_dim:

    # pretrain one layer
    autoencoder_next, encoder_next=pretrain_one_layer(autoencoder_cur, input_shape, current, x_train, x_test, epochs)

    # update the variables
    autoencoder_cur=autoencoder_next
    current=current//2

  return encoder_next # return the final encoder



# function to train the survival model
# finetune is true when we want to adapt the weights of the whole network, split_n is the number of splits we are using in cross-validation
def surv_training(x_train, t_train, x_test, t_test, encoder, finetune, epochs, split_n):

    # copy the encoder obtained from pretraining
    encoder_copy = keras.models.clone_model(encoder)
    encoder_copy.set_weights(encoder.get_weights())

    # fine-tune all layers
    if finetune:
        freeze_until = 0  # layer from which we want to fine-tune
        for layer in encoder_copy.layers[:freeze_until]:
            layer.trainable = False
    else:
        encoder_copy.trainable = False

    # architecture
    model = keras.Sequential()
    model.add(encoder_copy)
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.BatchNormalization())
    model.add(Dense(1))

    # data in the format required for the model
    time = np.concatenate((t_train, t_test))
    event = np.repeat(1, len(time))
    time_train = time[:len(t_train)]
    event_train = event[:len(t_train)]
    time_test = time[len(t_train):]
    event_test = event[len(t_train):]

    # training
    train_fn = InputFunction(x_train, time_train, event_train, drop_last=True, shuffle=True)
    eval_fn = InputFunction(x_test, time_test, event_test)

    trainer = TrainAndEvaluateModel(model=model, model_dir=Path("ckpts_" + str(split_n)), train_dataset=train_fn(),
                                    eval_dataset=eval_fn(), learning_rate=0.0001, num_epochs=epochs)
    trainer.train_and_evaluate()

    return model



# function to get the global SHAP values from the model
# kmeans_param is the number of sample with which we summarize the reference dataset
def explanation_algorithm(model, Dataset_cases_colonlung_controls, Dataset_controls_female, Dataset_controls_breast,
                          control_test, kmeans_param):

    # explanations with colon and lung cases and controls

    # define the algorithm
    explainer = shap.KernelExplainer(model, shap.kmeans(Dataset_cases_colonlung_controls, kmeans_param))

    # compute the local values
    shap_values = explainer.shap_values(x_test)

    # plot the global values for the split
    shap.summary_plot(shap_values, x_test, plot_type="bar")

    # save the SHAP global values
    shaps_cur=np.mean(np.abs(shap_values[0]), axis=0)

    # explanations with controls
    explainer1 = shap.KernelExplainer(model, shap.kmeans(Dataset_controls_female, kmeans_param))
    shap_values1 = explainer1.shap_values(x_test)
    shap.summary_plot(shap_values1, x_test, plot_type="bar")
    shaps1_cur=np.mean(np.abs(shap_values1[0]), axis=0)

    # explanations with breast controls
    explainer2 = shap.KernelExplainer(model, shap.kmeans(Dataset_controls_breast, kmeans_param))
    shap_values2 = explainer2.shap_values(x_test)
    shap.summary_plot(shap_values2, x_test, plot_type="bar")
    shaps2_cur=np.mean(np.abs(shap_values2[0]), axis=0)

    # explanations with matched controls
    explainer3 = shap.KernelExplainer(model, shap.kmeans(control_test, kmeans_param))
    shap_values3 = explainer3.shap_values(x_test)
    shap.summary_plot(shap_values3, x_test, plot_type="bar")
    shaps3_cur=np.mean(np.abs(shap_values3[0]), axis=0)

    # return the 4 types of SHAP values
    return shaps_cur, shaps1_cur, shaps2_cur, shaps3_cur



# function to compute the weighted Kendall-Tau distance
def normalised_kendall_tau_distance(values1, values2):

    # check the two lists
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)

    # count the weighted number of disordered pairs
    ndisordered = (np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])) / (
                i + 1) / (j + 1)).sum()

    # divide by the number of comparisons
    return ndisordered / (n * (n - 1))



# function to compute the KT-stability of the set of rankings
def KT_stab(shaps2, I):

    # initialize the results
    rank_shaps = np.empty(10 * 9 // 2)
    p = 0

    # for each couple of models
    for j in range(10):
        for k in range((j + 1), 10):
            cur = np.empty(10)
            cur1 = np.empty(10)
            l = 0

            # for each of the first top 10 features
            for i in I[:10]:

                # extract the SHAP values
                cur[l] = shaps2[j][i.astype(int)]
                cur1[l] = shaps2[k][i.astype(int)]
                l = l + 1

            # calculate the weighted KT distance and append it to the results
            dist = normalised_kendall_tau_distance(cur, cur1)
            rank_shaps[p] = dist
            p = p + 1

    # return the list of values (we will do the average afterwards)
    return rank_shaps



# function to calculate the mean of the importance values across models and the ranking
def mean_ranking(input_shape, importances):

    # calculate the mean importances across models
    mean_shaps = []
    for i in range(input_shape):
        cur = []
        for j in range(10):
            cur.append(importances[j][i])
        mean_shaps.append(np.mean(cur))

    # order the features according to the ranking
    mean_shaps1 = np.array(mean_shaps)
    I = np.argsort(-mean_shaps1)
    I = I.astype('str')
    z = -np.sort(-mean_shaps1)

    return I, z  # I is the final ranking, z are the ordered mean importance values



# function to train the survival model without pretraining
def surv_training_no_pretraining(x_train, t_train, x_test, t_test, input_shape, latent_dim, epochs, split_n):

    # input
    inputs = Input(shape=(input_shape,))
    cur_dim = input_shape // 2
    enc_intermediate = inputs

    # encoder
    while cur_dim >= latent_dim:
        enc_intermediate = Dense(cur_dim, activation='sigmoid')(enc_intermediate)
        enc_intermediate = keras.layers.BatchNormalization()(enc_intermediate)
        cur_dim = cur_dim // 2

    # last layers
    enc_intermediate = keras.layers.Dropout(0.1)(enc_intermediate)
    enc_intermediate = keras.layers.BatchNormalization()(enc_intermediate)
    latent = Dense(1)(enc_intermediate)
    model = Model(inputs, latent)

    # show the architecture
    model.summary()

    # data in the format required for the model
    time = np.concatenate((t_train, t_test))
    event = np.repeat(1, len(time))
    time_train = time[:len(t_train)]
    event_train = event[:len(t_train)]
    time_test = time[len(t_train):]
    event_test = event[len(t_train):]

    # training
    train_fn = InputFunction(x_train, time_train, event_train, drop_last=True, shuffle=True)
    eval_fn = InputFunction(x_test, time_test, event_test)

    trainer = TrainAndEvaluateModel(model=model, model_dir=Path("ckpts_" + str(split_n)), train_dataset=train_fn(),
                                    eval_dataset=eval_fn(), learning_rate=0.0001, num_epochs=epochs)
    trainer.train_and_evaluate()

    return model



# function to get the global SHAP values from the model using the breast controls reference dataset
def explanation_algorithm_breast(model, Dataset_controls_breast, kmeans_param, x_test):

    # explanations with breast controls
    explainer2 = shap.KernelExplainer(model, shap.kmeans(Dataset_controls_breast, kmeans_param))
    shap_values2 = explainer2.shap_values(x_test)
    shap.summary_plot(shap_values2, x_test, plot_type="bar")
    shaps2_cur=np.mean(np.abs(shap_values2[0]), axis=0)

    return shaps2_cur



# function to get the global SHAP values from the model using deep shap
def explanation_algorithm_deep(model, Dataset_cases_colonlung_controls, Dataset_controls_female, Dataset_controls_breast,
                          control_test):

    # explanations with colon and lung cases and controls

    # define the algorithm
    explainer = shap.DeepExplainer(model, Dataset_cases_colonlung_controls)

    # compute the local values
    shap_values = explainer.shap_values(x_test)

    # plot the global values for the split
    shap.summary_plot(shap_values, x_test, plot_type="bar")

    # save the SHAP global values
    shaps_cur=np.mean(np.abs(shap_values[0]), axis=0)

    # explanations with controls
    explainer1 = shap.DeepExplainer(model, Dataset_controls_female)
    shap_values1 = explainer1.shap_values(x_test)
    shap.summary_plot(shap_values1, x_test, plot_type="bar")
    shaps1_cur=np.mean(np.abs(shap_values1[0]), axis=0)

    # explanations with breast controls
    explainer2 = shap.DeepExplainer(model, Dataset_controls_breast)
    shap_values2 = explainer2.shap_values(x_test)
    shap.summary_plot(shap_values2, x_test, plot_type="bar")
    shaps2_cur=np.mean(np.abs(shap_values2[0]), axis=0)

    # explanations with matched controls
    explainer3 = shap.DeepExplainer(model, control_test)
    shap_values3 = explainer3.shap_values(x_test)
    shap.summary_plot(shap_values3, x_test, plot_type="bar")
    shaps3_cur=np.mean(np.abs(shap_values3[0]), axis=0)

    # return the 4 types of SHAP values
    return shaps_cur, shaps1_cur, shaps2_cur, shaps3_cur
