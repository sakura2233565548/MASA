
def get_pretraining(opts):

    from feeder.feeder_pretraining import Feeder_SLR
    training_data = Feeder_SLR(**opts.train_feeder_args)

    return training_data

