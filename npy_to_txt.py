import numpy as np
import cfg

if __name__ == "__main__":
    info = np.load(cfg.code_path+"/expert4_information.npy")
    np.savetxt(cfg.code_path+"/expert4_information_txt.txt", info)
    info = np.load(cfg.code_path+"/expert4_id.npy")
    np.savetxt(cfg.code_path+"/expert4_id_txt.txt", info)