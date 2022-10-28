class CFG():
    # model_path = '../pretrain_models/ernie3.0-chinese'
    model_path = '../../../pretrain_models/bert-base-chinese'
    train_path = "/root/pii_torch/data/corpus/train_0915.csv"
    # train_path = "//root/pii_torch/data/corpus/fudan/fudan_train.csv"
    # train_path = "/root/kaili/data/train_0813.csv"
    # val_path = "/root/pii_torch/data/corpus/fudan/fudan_test.csv"
    val_path = "../../data/corpus/dev_0915.csv"
    checkpoint_path = "./output"
    best_model_name = "best_model_prompt_samples_8.pt"
    label2id = {'其他': 0, '名字': 1, '地址': 2, '机构': 3}
    
    batch_size = 128
    epochs = 5
    max_len = 100
    num_lables = 14
    learn_rate = 2e-5
    gradient_accumulation_steps=1
    num_cycles=0.5
    num_warmup_steps=0
    max_grad_norm=1000
    schedule = "cosine"  # cosine
    eps = 1e-6
    betas= (0.9, 0.999)
    fgm = False
    prompt_type = "hard"  # na, hard, soft
    n_tokens = 8
    seed = 42

    device = "1"
    gpu = True
    # query = "以下是个人可识别信息的[MASK]信息："
    # query = "以下是描述个人可识别信息的[MASK]信息："
    # query = "名字信息包括中文名字，英文名字，中文拼音等。以下是个人可识别信息的[MASK]信息："
    # query = "地址信息很详细，包括中文地址和英文地址等。以下是个人可识别信息的[MASK]信息："
    query = "地址信息一般比较长，可以定位到具体地方。以下是个人可识别信息的[MASK]信息："
    # query = "地址信息包括省、市、街道，名字信息包括中文名字，英文名字，中文拼音等。以下是描述个人可识别信息的[MASK]信息："
    
