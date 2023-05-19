import timm

def timm_eva02_inat(pretrained=False, progress=True, device="cpu", **kwargs):
  model = timm.create_model("hf-hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21")

  return model


def timm_vitl_inat_dcomp(pretrained=False, progress=True, device="cpu", **kwargs):
  model = timm.create_model("hf-hub:timm/vit_large_patch14_clip_336.datacompxl_ft_inat21")

  return model


def timm_vitl_inat_laion(pretrained=False, progress=True, device="cpu", **kwargs):
  model = timm.create_model("hf-hub:timm/vit_large_patch14_clip_336.laion2b_ft_in12k_in1k_inat21")

  return model