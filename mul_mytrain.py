from deepsvg.mytrain import train
from deepsvg.config import _Config
from configs.deepsvg import myconfig

def main():
    cfg = myconfig.Config()

    model_name = "my_deepsvg"
    experiment_name = "my_experiment"
    log_dir = "./logs"
    debug = False
    resume = False
    train(cfg, model_name, experiment_name, log_dir, debug, resume)

if __name__ == '__main__':
    main()
