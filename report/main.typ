#import "sdust/lib.typ": *

#let title = [#v(-1.6em) Judge Me by My Size, Do You? Efficient Multi-Modal Learning for Mimir Using Pretrained Vision Models #v(1em)]

#show: submission.with(
  title: title,
  course: [#v(1em) AI508 Computer vision & AI509 Natural Language Processing],
  author: ("Simon Holm", "Johannes Rothe", "Shuagib Ibrahim"),
  group: "Learning Machines Learning Machine Learning",
  supervisor: ([Lukas Poech Galke], [Peter Schneider-Kamp]),
  date:   "Fall 2026",
  department:    imada,
  logo: image("IMADA_en.png", width:50%),
  info-width: 85%,
  outline-depth: 2,
)

#show link: set text(fill: rgb("#0000EE"))
#show link: underline




 

= Introduction
Recent progress in artificial intelligence has increasingly depended on scale, which puts the capable models out of reach for most researchers, companies and individuals. Mimir @Mimirv12026 tackles this very subject as a 1B parameter Hierarchical Reasoning Model @HRM2026. Mimir was trained on only 8 GPUs over a 3 week period, and being competitive with larger models on English and Danish benchmarks, it serves to show that models can be built without the need for the extreme scale of today's frontier model training.

Many modern models extend their capabilities beyond just text by becoming multi-modal @liu2023llava. While language models (including Mimir) perform great on text, they lack the ability to perceive and reason about visual inputs such as images. Vision models on the other hand are explicitly trained to perceive and contextualize images, but commonly fall short in reasoning. The field of Multi-modal Large Language Models (MLLMs) encapsulates these complementarities, extending what a single model can do @SurveyMLLM2026.

= Research question

== Research question
- How well can a pretrained language model, such as Mimir, be extended to process visual inputs by connecting it with a pretrained vision encoder?


- Can such Mimir be taught 2 concrete multimodal tasks #link("http://huggingface.co/datasets/allenai/pixmo-count")[pixmo-count], #link("http://huggingface.co/datasets/allenai/pixmo-cap-qa")[pixmo-cap-qa])?
- How does Mimir adapt to these vision-language tasks compared to a parameter-matched model, when both are trained under identical conditions. In terms of:
  - How fast does it reach peak accuracy?
  - Does Mimir reach a higher peak accuracy, using the same size constrained dataset?





==  Why it is interesting
The general paradigm in modern machine learning relies on _brute-force_ scaling, trying to have as much capacity as possible, and then learn complex tasks by seeing as much data as possible.
Mimir is moving in an opposite direction. By leveraging a better utilized and smaller model, so it can learn tasks and gain capabilities using smaller datasets. This has been demonstrated with Mimir v1.5 which learned 19 languages in just 12 hours, an incomprehensible achievement by today's standards. This project experiments in a completely different context, exploring if this efficiency is transferrable to vision.

This might help set a new, and better trend based on scaling innovation instead of parameters and data.

#pagebreak()
=  Experimental setup

==  Overall design
In this work, we make Mimir v1.5 @Mimirv152026 multi-modal by combining it with a pretrained vision encoder by prefixing the image encoding. We feature align Mimir to take input from the vision encoder by training an MLP. We then fine tune Mimir to for two conceptually distinct vision tasks. We then do the same for other models of parameter equivalent sizes, and measure if Mimir learns the task faster.
#figure(
 image("pipeline_example.png", width: 100%),
 caption: [Overview of the Mimir-SigLIP pipeline]
)


==  Datasets and models
We use SigLIP2 @SigLIP22025 and a 2-layer @liu2023llava15 MLP to project the encoder's patch features into Mimir v1.5's @Mimirv152026 embedding space.

We use the following datasets.

For pre-training:
- #link("http://huggingface.co/datasets/allenai/pixmo-cap")[pixmo-cap]: an image-caption dataset 
For instruction fine-tuning:
- #link("http://huggingface.co/datasets/allenai/pixmo-count")[pixmo-count]: an image-object-count dataset
- #link("http://huggingface.co/datasets/allenai/pixmo-cap-qa")[pixmo-cap-qa]: an image-question-and-asnwer dataset

We then compare the performance of Mimir with parameter matched models:
- HRM-Text 1B
- Ouro 1.4B
- Ouro 2.6B
- Qwen 3.5 4B
The specific on which of these, will depend on available compute.

#pagebreak()
==  Training
We train the projector on #link("http://huggingface.co/datasets/allenai/pixmo-cap")[pixmo-cap] for the feature alignment phase by freezing the vision and language model, and only using the gradients to update the MLP.
\Then we train Mimir to use the encoder features to perform two conceptually distinct VLM tasks, #link("http://huggingface.co/datasets/allenai/pixmo-count")[pixmo-count] and #link("http://huggingface.co/datasets/allenai/pixmo-cap-qa")[pixmo-cap-qa] with only the vision model freezed. All of this will be done with a LLM-as-a-Judge paradigm, using a judge from a different family to avoid same model bias. 

We then do the exact same thing for one or more different models.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: left,
    stroke: none,
    table.hline(stroke: 1.2pt),
    table.header([Stage], [Data], [SigLIP 2], [MLP], [Mimir]),
    table.hline(stroke: 0.6pt),
    [1 --- Alignment],  [PixMo-Cap],   [frozen], [trained], [frozen],
    [2a --- Counting],  [PixMo-Count], [frozen], [trained], [finetuned],
    [2b --- Visual QA], [PixMo-CapQA], [frozen], [trained], [finetuned],
    table.hline(stroke: 1.2pt),
    ),
  caption: [Training stages. Stage 2 consists of two separate tasks.],
  kind: table,
)

==  Evaluation

The training is measured as successful if the final product is a stable and reliable multimodal prototype, meaning it performs on a comparable level to an existing small vision-language model of similar size. The goal is not to outperform it, but to show competitive results with limited compute.


- The model architecture is successful when the SigLIP2 backbone pipeline with an MLP lets Mimir respond to picture inputs. 
- Successful when multimodal Mimir responds correctly, with an "LLM as a judge" reviewing its output on pixmo-cap-qa and giving each answer a verdict (correct or incorrect) against the reference answer.
- An evaluation against one or more comparable models, to see if Mimir performs better, earlier at previously mentioned tasks.

#pagebreak()
= Learning outcomes

This project combines NLP and Computer Vision by connecting a pretrained vision encoder to a pretrained language model.


- Implementing and fine-tuning a pretrained model using instruction tuning techniques, applied across two or more model architectures to demonstrate understanding of different design choices.

- Applying both NLP and computer vision principles to construct a multimodal model, using pre-existing model components such as SigLIP and Mimir to bridge language and visual representations, with established frameworks such as Hugging Face and PyTorch, drawing on methods and architectures covered in the course literature.

- Employing and comparing multiple model architectures against a defined baseline, evaluating performance across at least two different datasets to determine which configuration best addresses the research question.

- Fundamental concepts and Vision Transformers: applying SigLIP2, a Vision Transformer, and explaining how it turns images into patch embeddings.

- Implementing and fine-tuning vision models: building and training the vision pipeline.


#pagebreak()
#bib("refs.bib")

