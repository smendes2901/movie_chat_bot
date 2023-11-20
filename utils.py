import nlpaug.augmenter.char as nac
from random import choice 

def augment_string(string):
    augs = choice([nac.KeyboardAug(), nac.RandomCharAug(action="insert"), nac.RandomCharAug(action="substitute"), nac.RandomCharAug(action="swap"), nac.RandomCharAug(action="delete")])
    return augs.augment(string, n=1)[0]