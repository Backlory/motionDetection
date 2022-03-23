class GradScaler:
    def __init__(self, enabled = True):
        pass
    def scale(self, loss):
        return loss
    def step(self, opt):
        return opt
    def unscale_(self, opt):
        return opt
    def update(self):
        pass
    
class autocast:
    def __init__(self) -> None:
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args, **argkv):
        pass