   del argv

    try: 
        # init model class
        model = MbertPcnnModel(batch_size, dev_batch_size, 100, 1000,
                            args.bert_config_file,
                            args.learning_rate, num_train_steps, num_warmup_steps, args.use_tpu,
                            args.k1, args.k2, args.k3, args, init_checkpoint=None) # init_checkpoint=args.data_path+args.init_checkpoint

        tpu_cluster_resolver = None
        if args.use_tpu and args.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                args.tpu_name, zone=args.tpu_zone, project=None)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9


        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            session_config=config,
            cluster=tpu_cluster_resolver,
            master=None,
            model_dir=output_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=save_checkpoints_steps,
                num_shards=args.num_tpu_cores,
                per_host_input_for_training=is_per_host))     


        model_fn = eval("model.model_fn_v%s" % args.model_version)
        # create classifier
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=batch_size,
            eval_batch_size=dev_batch_size)

        # import pdb
        # pdb.set_trace()
        input_fn_trn = model.input_fn_builder([i_trn], is_training=True)
        input_fn_dev = model.input_fn_builder([i_dev], is_training=False)
        input_fn_tst = model.input_fn_builder([i_tst], is_training=False)

        # current_step = load_global_step_from_checkpoint_dir(output_dir)
        current_step = 1

        tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                        ' step %d.',
                        num_train_steps,
                        num_train_steps / (num_trn_example / batch_size),
                        current_step)

        start_timestamp = time.time()  # This time will include compilation time

        minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst, \
        maxci_step, maxci_dev, maxci_mse_dev, maxci_mse_tst, maxci_ci_tst = \
            restore_best_scores(0, best_model_dir_mse, best_model_dir_ci, estimator, input_fn_dev, input_fn_tst)

        last_time = start_timestamp
        while current_step < num_train_steps:
            next_checkpoint = min(current_step + save_checkpoints_steps, num_train_steps)
            estimator.train(input_fn=input_fn_trn, max_steps=next_checkpoint)

            checkpoint_time = int(time.time() - last_time)
            last_time = time.time()
            tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                            next_checkpoint, int(time.time() - start_timestamp))
            tf.logging.info('Starting to evaluate at step %d', next_checkpoint)

            import pdb
            pdb.set_trace()            
            eval_results = estimator.evaluate(input_fn=input_fn_dev, steps=dev_steps)
            tf.logging.info('Eval results at step %d: %s', next_checkpoint, eval_results)


            # minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst = \
            #     check_improvement_mse(minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst,
            #                         best_model_dir_mse, eval_results, current_step, estimator, input_fn_tst)

            # maxci_step, maxci_dev, maxci_mse_dev, maxci_mse_tst, maxci_ci_tst = \
            #     check_improvement_ci(maxci_step, maxci_dev, maxci_mse_dev, maxci_mse_tst, maxci_ci_tst,
            #                         best_model_dir_ci, eval_results, current_step, estimator, input_fn_tst)

            # info_scores(current_step, minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst,
            #             prefix='Current (sel_mse)',
            #             checkpoint_time=checkpoint_time)
            # info_scores(current_step, maxci_step, maxci_mse_dev, maxci_dev, maxci_mse_tst, maxci_ci_tst,
            #             prefix='Current (sel_ci)',
            #             checkpoint_time=checkpoint_time)
            # current_step = next_checkpoint


        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        num_train_steps, elapsed_time)

        # info_scores(current_step, minmse_step, minmse_dev, minmse_ci_dev, minmse_mse_tst, minmse_ci_tst, prefix='Final (sel_mse)')
        # info_scores(current_step, maxci_step, maxci_mse_dev, maxci_dev, maxci_mse_tst, maxci_ci_tst, prefix='Final(sel_ci)')
    except Exception as e: 
        print(e)
        import pdb
        pdb.set_trace()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run(main)
    main()
