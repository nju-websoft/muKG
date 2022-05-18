=================
Run models with main.py
=================

This is the python code to train TransE. This script shows how to select tasks and training mode::

    if __name__ == '__main__':
        t = time.time()
        curPath = os.path.abspath(os.path.dirname(__file__))
        model_name = 'transe'
        kg_task = 'lp'
        if kg_task == 'ea':
            args = load_args(curPath + "/args_ea/" + model_name + r"_args.json")
        elif kg_task == 'lp':
            args = load_args(curPath + "/args_kge/" + model_name + r"_args.json")
        else:
            args = load_args(curPath + "/args_et/" + model_name + r"_et_args.json")

        print(args.embedding_module)
        print(args)
        remove_unlinked = False
        if args.embedding_module == "RSN4EA":
            remove_unlinked = True
        kgs = read_kgs_from_folder(kg_task, args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                                   remove_unlinked=remove_unlinked)
        if kg_task == 'ea':
            model = ea_models(args, kgs)
        elif kg_task == 'lp':
            model = kge_models(args, kgs)
        else:
            model = et_models(args, kgs)
        model.get_model('TransE')
        model.run()
        model.test()
        print("Total run time = {:.3f} s.".format(time.time() - t))