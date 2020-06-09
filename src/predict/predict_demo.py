import tensorflow as tf
import pdb
# pdb.set_trace()
from src.finetune.dti_model import MbertPcnnModel
import argparse
import _pickle as cPickle
import os
import pdb

__author__ = 'Bonggun Shin'


tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='Training parser')
parser.add_argument('--gpu_num', default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7"], type=str)
parser.add_argument('--model_version', default="11", choices=["1", "2", "3", "4", "11", "14"], type=str)
parser.add_argument('--batch_size', default=512, choices=[256, 512], type=int)
parser.add_argument('--fold', default=0, choices=[0,1,2,3,4], type=int)
parser.add_argument('--dataset_name', type=str, default="kiba", choices=["davis", "kiba"],
                    help='dataset_name')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='learning_rate')
parser.add_argument('--tpu_name', type=str, default="btpu",
                    help='tpu_name')
parser.add_argument('--use_tpu', type=bool, default=False,
                    help='use_tpu')
parser.add_argument('--tpu_zone', type=str, default="us-central1-b",
                    help='tpu_zone')
parser.add_argument('--num_tpu_cores', type=int, default=8,
                    help='num_tpu_cores')
parser.add_argument('--bert_config_file', type=str, default="../../config/m_bert_base_config.json",
                    help='bert_config_file')
parser.add_argument('--init_checkpoint', type=str, default="../../data/elia/v11-no-atc/model.ckpt-750",
                    help='init_checkpoint')
parser.add_argument('--load-model-dir', type=str,help='model checkpoint dir')                    

parser.add_argument('--k1', type=int, default=12, help='kernel_size1')
parser.add_argument('--k2', type=int, default=12, help='kernel_size2')
parser.add_argument('--k3', type=int, default=12, help='kernel_size3')

parser.add_argument('--base_path', default="../../data", type=str)

# parser.add_argument('--use-atc', action='store_true', default=False)
# output_dir = './output'
parser.add_argument('--covid', action='store_true', default=False)
parser.add_argument('--use-atc', action='store_true', default=False)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

i_trn = "../../data/%s/tfrecord/fold%d.trn.tfrecord" % (args.dataset_name, args.fold)
i_dev= "../../data/%s/tfrecord/fold%d.dev.tfrecord" % (args.dataset_name, args.fold)
i_tst= "../../data/%s/tfrecord/fold%d.tst.tfrecord" % (args.dataset_name, args.fold)

if args.covid: 
    i_tst = '../../data/covid/tfrecord/fold0.tst.tfrecord'


if args.use_atc and args.covid: 
    filename = "atc-covid-predictions.txt"
elif args.use_atc and (not args.covid): 
    filename = "./atc-predictions.txt"
elif (not args.use_atc) and (not args.covid): 
    filename = "./baseline-kiba-predictions.txt"
else:
    raise Exception("filename invalid for parameters")


print("WILL WRITE TO: %s" % filename)

# output_dir = "../../data/%s/mbert_cnn_v%s_lr%.4f_k%d_k%d_k%d_fold%d/" % (args.dataset_name, args.model_version, args.learning_rate, args.k1, args.k2, args.k3, args.fold)


if args.dataset_name=="kiba":
    num_trn_example = 78835
    batch_size = args.batch_size
    # num_train_steps = 154000  # (78835/512)*1000 = 153974, 1000 epoch
    num_train_steps = int(num_trn_example*1.0/batch_size*1000)  # (78835/512)*1000 = 153974, 1000 epoch
    num_warmup_steps = num_train_steps//10
    dev_batch_size = 1
    dev_steps = 19709 # 19709/1=19709
    save_checkpoints_steps = 150 # 78835/512 = 157.67


elif args.dataset_name=="davis":
    num_trn_example = 20035
    batch_size = args.batch_size
    # num_train_steps = 40000  # (20035/512)*1000 = 39130, 1000 epoch
    num_train_steps = int(num_trn_example * 1.0 / batch_size * 1000)  # (78835/512)*1000 = 153974, 1000 epoch
    num_warmup_steps = num_train_steps // 10
    dev_batch_size = 5009
    dev_steps = 1 # 5009/1=5009
    save_checkpoints_steps = 40  # 20035/512 = 40.07

elif args.dataset_name=="metz":
    num_trn_example = 20035
    batch_size = args.batch_size
    num_train_steps = 12021  # (20035/500)*300 = 12, 300 epoch
    num_warmup_steps = num_train_steps // 10
    dev_batch_size = 5009
    dev_steps = 1 # 5009/1=5009
    save_checkpoints_steps = 158  # 20035/500 = 40.07
else:
    batch_size = None
    num_train_steps = None
    num_warmup_steps = None
    dev_batch_size = None
    dev_steps = None
    save_checkpoints_steps = None

def main(argv):
    del argv

    try: 
        # TODO: refactoring is required: seq_to_id.cpkl should be in one of the preprocessings
        refer_src = "%s/%s/seq_to_id.cpkl" % (args.base_path, args.dataset_name)
        if args.covid: 
            print("Using COVID cpkl file")
            lookup_file_name = "../../data/covid/seq_to_id.cpkl"
            
        with open(lookup_file_name, 'rb') as handle:
            (mseq_to_id, pseq_to_id) = cPickle.load(handle)

        with open(refer_src, 'rb') as handle:
            (refer_table, _) = cPickle.load(handle)            

        # os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # init model class
        model = MbertPcnnModel(batch_size, dev_batch_size, 100, 1000,
                            args.bert_config_file,
                            args.learning_rate, num_train_steps, num_warmup_steps, args.use_tpu,
                            args.k1, args.k2, args.k3, args,
                            use_atc=args.use_atc, 
                            pretrain_checkpoint=None, 
                            restore_checkpoint=args.load_model_dir, 
                            use_pretrain_init=False)

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
            model_dir=args.load_model_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=save_checkpoints_steps,
                num_shards=args.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        # import pdb
        # pdb.set_trace()

        # model_fn = eval("model.model_fn_v)
        # create classifier
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model.model_fn_v11,
            config=run_config,
            train_batch_size=batch_size,
            eval_batch_size=dev_batch_size)

        input_fn_tst = model.input_fn_builder([i_tst], is_training=False)

        summary = {}

        # 19708/4 = 4927
        print("====================================== tst ==============================")
        results = estimator.predict(input_fn=input_fn_tst)
        with open(filename, 'wt') as handle:
            # handle.write("chemid,pid,y_hat,y,smiles,fasta\n")
            handle.write("chemid,pid,y_hat,y\n")
            for idx, result in enumerate(results):

#                 (Pdb) xd_str
# '1,2,30,30,41,30,5,14,41,6,12,30,14,30,12,7,30,17,35,8,5,30,7,30,17,17,35,8,19,30,30,40,30,19,14,41,6,40,30,5,14,41,6,7,30,17,35,8,5,30,53,19,53,53,53,5,33,6,53,53,19,6,40,30,5,14,41,6,7,30,17,17,35,8,5,40,30,5,14,41,6,53,19,53,53,5,30,6,62,63,19,6,30,5,30,6,30,3,0,0'
# (Pdb) len(xd_str)

# 266
                len([x[:4] for x in mseq_to_id.keys() if x[:4][2]==','])
                tmp = '1,2,30,30,41' #no match on 11 characters
                search = [x for x in mseq_to_id.keys() if x.startswith(tmp)]
                xd_str = ','.join(map(str, result['xd'])) #[1,2,3,4,4,4] --> 1,2,3,4,4,4 --> f('1,2,3,4,4,4'): (smiles, chembl192381)
                xt_str = ','.join(map(str, result['xt']))

                import pdb
                pdb.set_trace()                
                
                refer_lens = [len(x) for x in refer_table.keys()]
                compare_lens = [len(x) for x in mseq_to_id.keys()]
                len_drug = len(xd_str)
                print("drug len: %d" % len_drug)

                if xd_str in mseq_to_id:
                    smiles = mseq_to_id[xd_str][0]
                    chemid = mseq_to_id[xd_str][1]
                else:
                    chemid = 0

                if xt_str in pseq_to_id:
                    fasta = pseq_to_id[xt_str][0]
                    pid = pseq_to_id[xt_str][1]
                else:
                    pid = 0

                y_hat = result['predictions'][0]
                y = result['gold'][0]

                # oneline = "%s,%s,%f,%f,%s,%s\n" % (chemid, pid, y_hat, y, smiles, fasta)
                oneline = "%s,%s,%f,%f\n" % (chemid, pid, y_hat, y)
                handle.write(oneline)
                # print(oneline)
                if idx % 1000 == 0:
                    # pdb.set_trace()
                    print(idx)
    except Exception as e: 
        print(e)
        pdb.set_trace()
    # print(idx)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
