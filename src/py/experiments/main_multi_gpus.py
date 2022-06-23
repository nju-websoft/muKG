import time
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
print(rootPath)
sys.path.append(rootPath)
#sys.path.append('/data1/xdluo/Knower')
from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.model.general_models import kge_models, ea_models, et_models
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == '__main__':
    t = time.time()
    curPath = os.path.abspath(os.path.dirname(__file__))
    model_name = 'conve'
    args = load_args(curPath + "/args_kge/" + model_name + r"_FB15K_args.json")

    print(args.embedding_module)
    print(args)
    remove_unlinked = False
    kgs = read_kgs_from_folder('lp', args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=remove_unlinked)
    model = kge_models(args, kgs)
    model.get_model('ConvE')
    model.run()
    model.test()
    print("Total run time = {:.3f} s.".format(time.time() - t))