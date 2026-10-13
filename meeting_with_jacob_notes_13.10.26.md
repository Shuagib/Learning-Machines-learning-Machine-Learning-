Simon:

Hrm bruger prefix attention.

trænings-kort
IMP200
PRO-6000

reducer compute i prompt igennem image tokens.

teste tasks uden at træne Mimir,
kun frys backbone SigLIP

- research question
sammenlign med en looped transformer
URO, N2B Qwen (kig på DFM huggingface)

Lav et testset (træn Mimir på et subset af data)
Mimir bør være bedere til at generelisere

- reasoning trace med gemma

slut 70+ er godt

start med peter


2k steps platou lr,then cosine annealing

Apply gemma 4s i transformers til at loade HMR-text


Shuagib:

HRM bidirectional attention
Slut 90'erne
Image classification, OCR,
1 Taks slut 80'erne accuracy 

Hiearchical reasoning model 
38-40 accuracy
kun trænet exact accuracy, måske ødelagt modellen fordi kun trænet på billeder 

Mange undersøge og få den til at lave:
- Hvis vi implementere det så får vi resultater
- end del Applikation studie on på hans kode
- RT6 kort 48 GB (kan han træne) 


- Prefix model: alle tokens kan attend til alle
- Lægge det foran teksten 



Hvad vi kan:
- Implementere det på enkel task.
- Prøve at gøre det hurtigere: Reducere compute (der er tricks).
- Du laver et netværk til at destillere hvad for nogle tokens den har (virker godt til billede classification).
- Start fra Peter's repo

tips:
- Træne det hele på samme tid
- Holde vision backbone parameter frozen
- Brug Sig-lip (flere open source visioner)
- Teste få task
- Vision backbone, SIP LIP (2 lag) (2 activation function) (mellem 2 valg muligheder)
- Subsample stille og roligt: e.g. 5000 for hver og subsample for 10000 hver og det skaler. Hvordan giver det mening at mixe dem.
- MIMIR skalar ikke som en LLM, man kan skalere og det virker (lidt ulækkert)
- Når vi træner den skal vi søger for at varme den op, 3e-4 over 2 steps og Cosine anneal over gulvtæppet. Behøver ikke at køre så langt fra der sker noget.
- Prøver at oversætte træning set til dansk.
- Vær opmærksom på hvor meget data i bruger i data settet (sampling).

- MIMIR virker ikke på VLLM fordi den tager gemma's 4 tokenizer. Vi skal gøre det i transformer's. Gøre det måske lidt langsommere. Vi skal have en transformer vision, som kører den.

MIMIR:
- Hvordan ændrer det sig når vi har en Mindre model, som klarer sig godt på sprog, hvordan kan den klare sig på billedere. 

sammenligne:
- UURO/ N2B-Gwen model som bench mark: Backbone og sammenlign (sanitet check). Når vi træner begge two, hvordan skalerer vi vores performance (loss, validation and test set). Forventer at MIMIR klarer sig på data den ikke har set på modellerne. Dette demonterer HRM forstår mere hvordan der foregår. 



metric:
- Accuracy slut 70'erne. Kommer vi højere skal vi snakke om scalering 


ekstra ting:
- Add reasoning trace
- 


Johannes:
- han har trænet ocr, image classiication og reasoning
- da han kun gjorde det med en task havde en en acc p 80, men med 4 tasks har han slut 90
- han ikke vedligeholdt text delen
- prefix model betyder at alt information i prompt er set som en encoder model, så den kan kigge frem og tilbage (så tekst kan arbejde med billedet)
- vi kunne arbejde med at effektivisere det, så vi minimere hvor mange tokens vi reservere til billedet
- vi skal træne mimir og mlp sammen, men holde vision backbone frosset
- subsample taskene fordi de er kæmpestore
- bare siglip
    - hare flere opensource versioner
- vi skal argumentere for at 2 mlp lag er bedre end en (Simon har læst en artikel)
- sammenligne med en anden transformer (der ikke har sparse attention) ouroro fra bytedance med loop. eller 2b QWEN model. tjek ordbogens tech rapporter og tag en af dem de sammenligner med
    - Sammenlign skalering med loss for at mimir skal skalere hurtigere
    - sammenlign om mimir hurtigere generaliserer til et test set
- vi skal overveje om der skal være reasonning traces (den reasoner noramlt i latent, men det hælper med tokens)
- vi skal have en training acc på 70, og alt over er godt. hvis ikke, snak med ham fordi det kunne bare være skalering
- vi skal starte fra peters repo
    - https://github.com/schneiderkamplab/HRM-Text
    - (ikke tag submoduler)
    - varme den op 3e-4 over 2k steps, 2k steps med linear, og cosine annealing efterfølgende
    - han har ikke fået gains efter 5000 steps, der rammer den et plateu
- Først træner den på engelsk og så prøver at oversætte det til dansk og se hvor hurtigt den kan oversætte læringen fra engelsk til dansk
- Det vigtigste er hvor meget data vi bruger, og hvor lidt compute vi kan bruge
- apply gemma4 tokenizer inden vi giver den til mimir
