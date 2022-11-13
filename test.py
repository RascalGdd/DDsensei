from models.models import cfg
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
from dataloaders.synthia import synthia_dataloader

#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
# dataloader_val = dataloaders.get_dataloaders(opt)
dataloader_val = synthia_dataloader
#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
model = models.Unpaired_model(opt, cfg)
model = models.put_on_multi_gpus(model, opt)
model.eval()

#--- iterate over validation set ---#
for i, data_i in enumerate(dataloader_val):
    _, label = models.preprocess_input(opt, data_i)
    generated = model(_, label, "generate", None)
    image_saver(label, generated, data_i["name"])
