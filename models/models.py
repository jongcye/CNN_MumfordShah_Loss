
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'unet':
        from .unet_model import UNetModel
        model = UNetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
