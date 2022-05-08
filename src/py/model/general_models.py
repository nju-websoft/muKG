import os

from src.py.evaluation.evaluation import LinkPredictionEvaluator
from src.py.util.env_checker import module_exists


class ModelFamily_tf(object):
    def __init__(self, args, kgs):
        self.args = args
        self.kgs = kgs

    def infer_model(self, model_name):
        from src.tf.kge_models.kge_trainer import kge_trainer
        # from src.tf.kge_models.transd import TransD

        from src.tf.ea_models.bootea import BootEA
        from src.tf.ea_models.rdgcn import RDGCN
        from src.tf.ea_models.mtranse import MTransE
        from src.tf.ea_models.attre import AttrE
        from src.tf.ea_models.sea import SEA
        from src.tf.ea_models.imuse import IMUSE
        from src.tf.ea_models.gcn_align import GCN_Align
        from src.tf.ea_models.iptranse import IPTransE
        from src.tf.ea_models.jape import JAPE

        if model_name == 'AttrE':
            return AttrE()
        elif model_name == 'BootEA':
            return BootEA()
        elif model_name == 'GCN_Align':
            return GCN_Align()
        elif model_name == 'IMUSE':
            return IMUSE()
        elif model_name == 'JAPE':
            return JAPE()
        elif model_name == 'MTransE':
            return MTransE()
        elif model_name == 'IPtransE':
            return IPTransE()
        elif model_name == 'RDGCN':
            return RDGCN()
        elif model_name == 'SEA':
            return SEA()

        from src.tf.kge_models.Analogy import Analogy
        from src.tf.kge_models.TransE import TransE
        from src.tf.kge_models.TransH import TransH
        """
        from src.tf.kge_models.ComplEx import ComplEx
        from src.tf.kge_models.DistMult import DistMult
        from src.tf.kge_models.HolE import HolE
        from src.tf.kge_models.RESCAL import RESCAL
        from src.tf.kge_models.TransD import TransD
        from src.tf.kge_models.TransR import TransR
        """
        # self.TransD = TransD()
        '''
        self.TransH = TransH(kgs, args)
        self.TransE = TransE(kgs, args)
        self.HolE = HolE(kgs, args)
        self.Analogy = Analogy(kgs, args)
        self.RESCAL = RESCAL(kgs, args)
        self.ComplEx = ComplEx(kgs, args)
        self.DistMult = DistMult(kgs, args)
        self.TransR = TransR(kgs, args)
        '''
        if model_name == 'Analogy':
            return Analogy(self.kgs, self.args)
        elif model_name == 'TransE':
            return TransE(self.kgs, self.args)
        elif model_name == 'TransH':
            return TransH(self.kgs, self.args)
        """
        elif model_name == 'TransD':
            return TransD(self.args, self.kgs)
        elif model_name == 'TransR':
            return TransR(self.kgs, self.args)
        elif model_name == 'HolE':
            return HolE(self.kgs, self.args)
        elif model_name == 'RESCAL':
            return RESCAL(self.kgs, self.args)
        elif model_name == 'ComplEx':
            return ComplEx(self.kgs, self.args)
        elif model_name == 'DistMult':
            return DistMult(self.kgs, self.args)
        """


class ModelFamily_torch(object):
    def __init__(self, args, kgs):
        self.args = args
        self.kgs = kgs

    def infer_model(self, model_name):
        from src.torch.kge_models.Analogy import Analogy
        from src.torch.kge_models.ComplEx import ComplEx
        from src.torch.kge_models.DistMult import DistMult
        from src.torch.kge_models.HolE import HolE
        from src.torch.kge_models.RESCAL import RESCAL
        from src.torch.kge_models.TransD import TransD
        from src.torch.kge_models.TransE import TransE
        from src.torch.kge_models.TransH import TransH
        from src.torch.kge_models.TransR import TransR
        from src.torch.kge_models.RotatE import RotatE
        #from src.torch.kge_models.SimplE import SimplE
        from src.torch.kge_models.TuckER import TuckER
        from src.torch.kge_models.ConvE import ConvE
        from src.torch.ea_models.trainer.attre_trainer import attre_trainer
        from src.torch.ea_models.trainer.bootea_trainer import bootea_trainer
        from src.torch.ea_models.trainer.gcn_align_trainer import gcn_align_trainer
        from src.torch.ea_models.trainer.imuse_trainer import imuse_trainer
        from src.torch.ea_models.trainer.iptranse_trainer import iptranse_trainer
        from src.torch.ea_models.trainer.jape_trainer import jape_trainer
        from src.torch.ea_models.trainer.mtranse_trainer import mtranse_trainer
        from src.torch.ea_models.trainer.rdgcn_trainer import rdgcn_trainer
        from src.torch.ea_models.trainer.sea_trainer import sea_trainer
        from src.torch.et_models.TransE_ET import TransE_ET
        from src.torch.et_models.RESCAL_ET import RESCAL_ET
        from src.torch.et_models.HolE_ET import HolE_ET
        if model_name == 'TransE':
            return TransE(self.kgs, self.args)
        if model_name == 'TransE_ET':
            return TransE_ET(self.kgs, self.args)
        elif model_name == 'RESCAL_ET':
            return RESCAL_ET(self.args, self.kgs) 
        elif model_name == 'HolE_ET':
            return HolE_ET(self.args, self.kgs) 
        elif model_name == 'TransD':
            return TransD(self.args, self.kgs)
        elif model_name == 'TransH':
            return TransH(self.kgs, self.args)
        elif model_name == 'TransR':
            return TransR(self.kgs, self.args)
        elif model_name == 'HolE':
            return HolE(self.kgs, self.args)
        elif model_name == 'Analogy':
            return Analogy(self.kgs, self.args)
        elif model_name == 'RESCAL':
            return RESCAL(self.kgs, self.args)
        elif model_name == 'ComplEx':
            return ComplEx(self.kgs, self.args)
        elif model_name == 'DistMult':
            return DistMult(self.kgs, self.args)
        elif model_name == 'RotatE':
            return RotatE(self.args, self.kgs)
        #elif model_name == 'SimplE':
        #    return SimplE(self.args, self.kgs)
        elif model_name == 'TuckER':
            return TuckER(self.args, self.kgs)
        elif model_name == 'ConvE':
            return ConvE(self.args, self.kgs)
        # load ea models
        if model_name == 'AttrE':
            return attre_trainer()
        elif model_name == 'BootEA':
            return bootea_trainer()
        elif model_name == 'GCN_Align':
            return gcn_align_trainer()
        elif model_name == 'IMUSE':
            return imuse_trainer()
        elif model_name == 'JAPE':
            return jape_trainer()
        elif model_name == 'MTransE':
            return mtranse_trainer()
        elif model_name == 'IPtransE':
            return iptranse_trainer()
        elif model_name == 'RDGCN':
            return rdgcn_trainer()
        elif model_name == 'SEA':
            return sea_trainer()
        # self.SimplE = SimplE(kgs, args)


class kge_models:
    """
        kgs need to be rewriten,  its a single kg
        3.11 tf code needs to be suplemented
    """
    def __init__(self, args, kgs):
        self.model = None
        self.args = args
        self.kgs = kgs

    def get_model(self, model_name):
        if module_exists():
            from src.torch.kge_models.kge_trainer import kge_trainer, parallel_trainer
            mf = ModelFamily_torch(self.args, self.kgs)
            self.args.is_torch = True
            mod = mf.infer_model(model_name)
            if self.args.is_parallel:
                self.model = parallel_trainer()
            else:
                self.model = kge_trainer()
            self.model.init(self.args, self.kgs, mod)
        else:
            from src.tf.kge_models.kge_trainer import kge_trainer
            mf = ModelFamily_tf(self.args, self.kgs)
            self.args.is_torch = False
            mod = mf.infer_model(model_name)
            self.model = kge_trainer()
            self.model.init(self.args, self.kgs, mod)

    def test(self):
        self.model.retest()

    def save(self):
        self.model.save()

    def run(self):
        self.model.run()


class ea_models:
    """
        kgs need to be rewriten,  its a single kg
        3.11 tf code needs to be suplemented
    """
    def __init__(self, args, kgs):
        self.model = None
        self.args = args
        self.kgs = kgs

    def get_model(self, model_name):
        if module_exists():
            mf = ModelFamily_torch(self.args, self.kgs)
            self.args.is_torch = True
            self.model = mf.infer_model(model_name)
            self.model.init(self.args, self.kgs)
        else:
            mf = ModelFamily_tf(self.args, self.kgs)
            self.args.is_torch = False
            self.model = mf.infer_model(model_name)
            self.model.set_args(self.args)
            self.model.set_kgs(self.kgs)
            self.model.init()
            #self.model.run()

    def test(self):
        self.model.retest()

    def save(self):
        self.model.save()

    def run(self):
        self.model.run()


class et_models:
    """
        kgs need to be rewriten,  its a single kg
        3.11 tf code needs to be suplemented
    """
    def __init__(self, args, kgs):
        self.model = None
        self.args = args
        self.kgs = kgs

    def get_model(self, model_name):
        if module_exists():
            from src.torch.et_models.et_trainer import et_trainer
            mf = ModelFamily_torch(self.args, self.kgs)
            self.args.is_torch = True
            mod = mf.infer_model(model_name)
            self.model = et_trainer()
            self.model.init(self.args, self.kgs, mod)
        else:
            mf = ModelFamily_tf(self.args, self.kgs)
            self.args.is_torch = False
            self.model = getattr(mf, model_name)
            self.model.set_args(self.args)
            self.model.set_kgs(self.kgs)
            self.model.init()
            self.model.run()

    def test(self):
        self.model.retest()

    def save(self):
        self.model.save()

    def run(self):
        self.model.run()
