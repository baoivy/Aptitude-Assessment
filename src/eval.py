import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse
import json
from preprocessing import ReviewDataset
from model import ScoreReviewModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_args():
    #rewrite this 
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='/content/drive/MyDrive/AISIA-Assignment/lightning_logs/version_24/checkpoints/epoch=0-step=576.ckpt')
    parser.add_argument("--input_file", type=str, default='data/test_dataset.json')
    parser.add_argument("--output_file", type=str, default='public_test.json')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model", type=str, default="xlm-roberta-base")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--predict", default=True, type=bool)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()

def evaluate(result, ground_truth):
	"""
		Evaluates the predicted classes w.r.t. a gold file.
		Metrics are: f1-macro, f1-micro and accuracy
		Remove precheck format on this file
		:param pred_fpath: a json file with predictions,
		:param gold_fpath: the original annotated gold file.
	"""

	print(classification_report(ground_truth, result, target_names=['class 1', 'class 2', 'class 3', 'class 4', 'class 5']))
	cm = confusion_matrix(ground_truth, result, labels=[0,1,2,3,4])
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
	disp.plot()
	plt.show()



def get_gold_label(input_file):
	with open(input_file, 'r', encoding='utf-8') as f:
			data = json.load(f)
	return [int(float(element['score'])) - 1 for element in data]

def main():
	args = get_args()
	model = ScoreReviewModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
	model.eval()
	datas = ReviewDataset(args=args)
	pin_memory = True if args.num_workers > 0 else False
	test_dataloader = DataLoader(datas, batch_size=args.batch_size,
								shuffle=False)
												#pin_memory=pin_memory)

	trainer = pl.Trainer(accelerator="gpu")
	results = trainer.predict(model, test_dataloader)
	goal_label = get_gold_label(args.input_file)
	results_unbatch = [res for batch in results for res in batch]
	evaluate(results_unbatch, goal_label)
						

if __name__ == '__main__':
		main()