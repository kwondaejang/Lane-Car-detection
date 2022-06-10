import torch
import torchvision
import os


# save model
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# load model
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


# function to run prediction on image and save
def save_predictions_as_imgs(loader, model, epoch, device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if(idx % 100 != 0):
            continue
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            #preds = (model(x))
            preds = (preds > 0.5).float()

        path = os.path.join("saved_images", "epoch_{}".format(epoch))
        os.makedirs(path, exist_ok=True)
        torchvision.utils.save_image(preds, os.path.join(path, "pred_{}.png".format(idx)))
        torchvision.utils.save_image(y, os.path.join(path, "true_{}.png".format(idx)))