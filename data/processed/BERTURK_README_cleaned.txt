BERTurk

 
 
 

! DOI.
 25.03.2020: Release of BERTurk uncased model and BERTurk models with larger vocab size (128k, cased and uncased).
 11.03.2020: Release of the cased distilled BERTurk model: DistilBERTurk .
 Available on the Hugging Face model hub we can train both cased and
uncased models on a TPU v3-8. You can find the TensorBoard outputs for
the training here:

 TensorBoard cased model.

A detailed cheatsheet of how the models were trained, can be found here.

C4 Multilingual dataset (mC4)

We've also trained an ELECTRA (cased) model on the recently released Turkish part of the
multiligual C4 (mC4) corpus.

Turkish Model Zoo

Here's an overview of all available models, incl. their training corpus size:

 Model hub link 
 
 here here here here here here 262GB 

DistilBERTurk

The distilled version of a cased model, so called DistilBERTurk, was trained
on 7GB of the original training data, using the cased version of BERTurk
as teacher model.

DistilBERTurk was trained with the official Hugging Face implementation from
here.

ELECTRA

In addition to the BERTurk models, we also trained ELECTRA small and base models. A detailed overview can be found
in the ELECTRA section.

ConvBERTurk

In addition to the BERT and ELECTRA based models, we also trained a ConvBERT model. The ConvBERT architecture is presented
in the "ConvBERT: Improving BERT with Span-based Dynamic Convolution".

mC4 ELECTRA

In addition to the ELECTRA base model, we also trained an ELECTRA model on the Turkish part of the mC4 corpus. We use a
sequence length of 512 over the full training time and train the model for 1M steps on a v3-32 TPU.

BERT5urk

BERT5urk is a new 1.42B encoder-decoder model based on the efficient.

All evaluations are performed with the awesome Flair library and the evaluation code and configs can be found in the
bs8-e3-lr5e-05bs16-e3-lr5e-05bs8-e3-lr5e-05bs8-e3-lr5e-05bs16-e3-lr5e-05bs8-e3-lr5e-05bs16-e10-lr3e-05bs8-e10-lr3e-05bs8-e10-lr3e-05bs8-e10-lr5e-05bs16-e10-lr3e-05bs8-e10-lr5e-05 16, 8 3e-5, 5e-5 3 bs16-e3-lr3e-05bs16-e3-lr3e-05bs16-e3-lr3e-05bs16-e3-lr3e-05bs8-e3-lr3e-05bs16-e3-lr5e-05 

Acknowledgments

Thanks to Kemal Oflazer ( for providing us
additional large corpora for Turkish. Many thanks to Reyyan Yeniterzi for providing
us the Turkish NER dataset for evaluation.

We would like to thank Merve Noyan ( for the
awesome logo!

Research supported with Cloud TPUs from the awesome TRC program (

Many thanks for providing access to the TPUs over a lot of years