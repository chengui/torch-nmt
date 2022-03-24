import os
from torch.utils.data import Dataset


class TatoebaDataset(Dataset):
    def __init__(self, root_dir='.', split='train', lang_pair=('eng', 'fra'),
                 processor=None, num_samples=None, test_ratio=0.1):
        self.src_lang, self.tgt_lang = lang_pair
        self.src_text, self.tgt_text = self._read_text(root_dir, lang_pair)
        if num_samples is not None:
            self.src_text = self.src_text[:num_samples]
            self.tgt_text = self.tgt_text[:num_samples]

        if isinstance(processor, (list, tuple)):
            src_processor, tgt_processor = processor
        else:
            src_processor, tgt_processor = processor, processor
        src_data, self.src_vocab = src_processor(self.src_text)
        tgt_data, self.tgt_vocab = tgt_processor(self.tgt_text)

        test_samples = int(len(src_data) * test_ratio)
        if split == 'train':
            self.src_data = src_data[:-test_samples]
            self.tgt_data = tgt_data[:-test_samples]
        else:
            self.src_data = src_data[test_samples:]
            self.tgt_data = tgt_data[test_samples:]

    def __getitem__(self, idx):
        return (self.src_data[idx], self.tgt_data[idx])

    def __len__(self):
        return len(self.src_data)

    def _read_text(self, root_dir, lang_pair):
        src_lang, tgt_lang = lang_pair
        filename = '{1}-{0}/{1}.txt'.format(src_lang, tgt_lang)
        filepath = os.path.join(root_dir, filename)
        src_text, tgt_text = [], []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                src, tgt = line.strip().split('\t')
                src_text.append(src)
                tgt_text.append(tgt)
        return src_text, tgt_text
