def train_and_evaluate(train_data_pattern, eval_data_pattern, test_data_pattern, export_dir, output_dir):
    ...
    train_dataset = read_dataset(train_data_pattern, train_batch_size)
    eval_dataset = read_dataset(eval_data_pattern, eval_batch_size, tf.estimator.ModeKeys.EVAL, num_eval_examples)

    model = create_model()
    history = model.fit(train_dataset,
                        validation_data=eval_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[cp_callback])
    # export
    logging.info('Exporting to {}'.format(export_dir))
    tf.saved_model.save(model, export_dir)

