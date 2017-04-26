Setup
Download the corpus with:

    ./download.sh

Then run `dataset_translate.py` to generate text files for the desired languages.

Example Usage

To learn to translate from English to German:

        python main.py --encoder-input-filename english_to_german.txt --decoder-input-filename german_to_english.txt --encoder-weights encoder_english.h5 --decoder-weights decoder_deutsch.h5
