from argparse import ArgumentParser
from IQA.iqa import IQA
from pathlib import Path


def main(img_path: str = None):
    iqa = IQA()
    if img_path is None:
        img_path = Path('mock_images', 'mock.jpg')
    absolute_path = img_path.resolve()

    predition = iqa.predict(absolute_path)
    print('Predicted score:', predition)
    issue = iqa.find_issues(predition, str(absolute_path))
    print('Issues:', issue)


if __name__ == '__main__':
    img_path = None
    argparser = ArgumentParser()
    argparser.add_argument('--image_path', type=str, required=False)
    args = argparser.parse_args()

    if args.image_path:
        img_path = args.image_path
    main(img_path)
