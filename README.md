# Análisis de Sentimientos mediante Redes neuronales
- Universidad Politécnica
- Máster Universitario en Inteligencia Artificial
- Web Science
- Enero 2020

## Autores
- Antonio Sejas Mustafá
- Elena Saa Noblejas

# INTRODUCCIÓN

El objetivo de esta práctica es realizar una aplicación implementando alguno de los métodos vistos en clase. Por tanto tuvimos que elegir entre Sistema de recomendación, Clasificación de documentos siguiendo Topic Models, Reconocimiento de Entidades o Análisis de Sentimiento.

Nosotros decidimos desarrollar este último proyecto. De este modo, nuestro objetivo es realizar una aplicación que dado un texto sea capaz de identificar si caracter positivo o negativo.

Más concretamente hemos decidido trabajar sobre un dataset ya conocido. El dataset de reviews de películas de IMDB. En el siguiente apartado ampliamos la información sobre el dataset y comentamos dónde está disponible para su descarga.

Nuestros textos, como ya hemos comentado son reviews de películas, y el sentimiento será si una review le ha gustado a un usuario o no.

**¿Pero cómo vamos a evaluar nuestra aplicación?**

Un texto puede estar lleno de ambiguedad, incluso una misma review puede tener comentarios pariales de caracter positivo y otras críticas negativas. Nosotros vamos a ignorar estas situaciones y seguiremos el gold standard marcado por el artículo descrito en Maas, A. L. et. al. 2011 [1]

En este artículo se describe que las reviews positivas son aquellas que tengan una nota de 7 estrellas o más. Y de forma equivalente las negativas son las que tengan asociada una valoración de 0 a 4 estrellas. De esta forma no se tienen en cuenta los textos más ambiguos o indecisos, las reseñas Neutrales.


## El Dataset de IMDB
Este dataset ha sido realizado por los investigadores de Stanford autores del artículo original [1] .

El dataset original está disponible en: http://ai.stanford.edu/~amaas/data/sentiment/

El dataset cuenta con 50.000 reviews, que ellos utilizaron 25.000 para entrenamiento y 25.000 para testing. Nosotros para reducir complejidad solo usaremos una mitad, que hemos tratado previamente para eliminar caracteres raros y poner los textos en minúsculas.

Además este dataset cuenta con un porcentaje equilibrado de reseñas. Siendo la mitad positivas y la otra mitad negativas.

Cada review tiene una etiqueta que lo categoriza de positiva o negativamente.


## Tecnologías utilizadas

A continuación describimos la metodología que hemos seguido para el analizador de sentimientos. Cada uno de estos puntos corresopnde con una sección del código.

**Limpieza del dataset**

El primer paso es analizar y hacer un tratamiento de los textos del dataset. En el área de procesamiento del lenguaje natural hay una gran cantidad de alternativas y posibilidades. Es posible realizar distintas representación de los textos del corpus, y una gran variedad de extracción de características.

El planteamiento del analizador de sentimientos puede verse de alguna forma con un clasificador de textos, en el que se intenta clasificar un text (review) como positivo o negativo.

Por este motivo todos los métodos de representación utiizados en PLN son válidos. Algunos de estos modelos son: bag of words, vector space model, tf-idf, topic model.

Nosotros hemos decidido obtener una bolsa de palabras (bag of words), teniendo en cuenta la frecuencia relativa con respecto a la otra clase. La idea es similar a un TF-IDF pero a nivel de clase. Esta representación nos permitirá darle más peso a las palabras más polarizadas.

Por mantener determinar un límite en esta práctica no aplicaremos ningún procedimiento lematización ni stemming. Tampoco utilizaremos n-gramas. Únicamente eliminaremos las palabras vacías, stopwords, para reducir el ruido de los datos. La tokenización utilizada consiste en convertir las palabas en índices de un array. Todo este tipo de técnicas las hemos visto en clase y también se describen en más detalle en el libro"Natural Language Processing in Action" [3]

**Entrenamiento**

De forma similar a la limipieza del dataset y la extracción de variables predictoras, en el entrenamiento podemos utilizar prácticamente cualquier algoritmo de clasificación. Desde un Naive Bayes, Support Vector Machine, árboles de decisión o redes neuronales son algunas de las opciones más utilizadas.

Nosotros al no haber cursado ninguna asignatura de redes neuronales, hemos decidido utilizar una red neuronal, en concreto un Perceptrón multicapa.

La primera capa, capa de entrada, tendrá tantos nodos como palabras haya en nuestro vocabulario. La segunda capa tendrá 30 nodos, de forma experimental hemos observado un buen comportamiento con 10 a 30 nodos.

Por último la capa de salida tendrá un solo nodo que dará un valor comprendido entre 0 y 1. Cuanto más cerca del 1 , más positiva se considerará la reseña. Un valor cercano al 0.5 se considerará la reseña "neutral".


**Validación**

Por último, nosotros hemos preferido evaluar la precisión de nuestro algoritmo utilizando un dropout 70/30 por sencillez de implementación. Una solución más profesional requeriría utilizar métodos de validación más sofistiados como un k-fold.

Además hemos observado que incluso usando la mitad del dataset de entrenamiento, obtenemos valores muy cercanos al SVM descrito en el artículo [1]. En el artículo se alcanzan precisiones de entorno al 0.88, mientras que como veremos nuestra red se queda en 0.86 debido a falta de reducción de ruido comentada anteriormente.

**Extra**

De forma adicional, hemos creado una celda con una caja de texto para que se pueda comprobar el fucionamiento con textos fuera del dataset. Esta caja de texto está identificada bajo el título "Inserta un texto para probar el analizador de sentimientos". Hay que escribir un texto y ejecutar esa celda y la siguiente para ver los resultados.

- El código está autocontenido en este Jupyter Notebook. El cual está disponible online: https://colab.research.google.com/drive/11ZvUGrctfSbuTqa_tk3J-SNkEolqmZbK
- Además el código fuente y el dataset están disponibles en Github: https://github.com/sejas/muia-imdb-sentiment-analysis


## Referencias

1. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150). Association for Computational Linguistics. [Descargar Artículo](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)

2. Hochreiter, Sepp & Schmidhuber, Jürgen. (1997). Long Short-term Memory. Neural computation. 9. 1735-80. 10.1162/neco.1997.9.8.1735. 

3. Lane, H., Howard, C., & Hapke, H. M. (2019). Natural Language Processing in Action: Understanding, Analyzing, and Generating Text with Python. Manning Publications Company.


# CARGA DEL DATASET

```python
# Importar librerías
import sys
import pandas as pd
import numpy as np
from collections import Counter
```


```python
# Carga de los datos en un dataframe
df = pd.read_csv('imdb.csv', index_col=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>labels</th>
      <th>reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POSITIVE</td>
      <td>bromwell high is a cartoon comedy . it ran at ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NEGATIVE</td>
      <td>story of a man who has unnatural feelings for ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POSITIVE</td>
      <td>homelessness  or houselessness as george carli...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NEGATIVE</td>
      <td>airport    starts as a brand new luxury    pla...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POSITIVE</td>
      <td>brilliant over  acting by lesley ann warren . ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    25000



Los datos de `imdb.csv` han sido preprocesados y el contenido está preparado para contener solo caracteres en minúsculas. Esto es para simplificar el la identificación de las palabras, independientemente de cómo hayan sido escritas.


# ANÁLISIS Y TRATAMIENTO PREVIO DEL DATASET

Utilizando tres objetos `Counter` podemos calcular la frequencia absoluta para cada tipod e clase, positiva y negativa y un tercer counter para la contabilizar la frecuencia total de cada palabra en el corpus.


```python
positive_freq = Counter()
negative_freq = Counter()
total_freq = Counter()
```

Además de contabilizar la frecuencia de cada palabra en cada clase y en total, aprovechamos para eliminar las palabras vacías previamente conocidas y facilitadas por sklearn.


```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
custom_stopwords = stop_words.ENGLISH_STOP_WORDS.union(['br', '.'])
def remove_stopwords(text):
  return [word for word in text.split(' ') if word not in custom_stopwords]
```


```python
for _, (label, review) in df[df['labels']=='POSITIVE'].iterrows():
  positive_freq += Counter(remove_stopwords(review))
for _, (label, review) in df[df['labels']=='NEGATIVE'].iterrows():
  negative_freq += Counter(remove_stopwords(review))

total_freq = positive_freq + negative_freq


```

Extraemos las palabras de las reseñas positivas y negativas ordenándolas de más a menos comunes.


```python
positive_freq.most_common(20)
```




    [('', 550468),
     ('s', 33815),
     ('film', 20937),
     ('movie', 19074),
     ('t', 13720),
     ('like', 9038),
     ('good', 7720),
     ('just', 7152),
     ('story', 6780),
     ('time', 6515),
     ('great', 6419),
     ('really', 5476),
     ('people', 4479),
     ('best', 4319),
     ('love', 4301),
     ('life', 4199),
     ('way', 4036),
     ('films', 3813),
     ('think', 3655),
     ('movies', 3586)]




```python
negative_freq.most_common(20)
```




    [('', 561462),
     ('s', 31546),
     ('movie', 24965),
     ('t', 20361),
     ('film', 19218),
     ('like', 11238),
     ('just', 10619),
     ('good', 7423),
     ('bad', 7401),
     ('really', 6262),
     ('time', 6209),
     ('don', 5336),
     ('story', 5208),
     ('people', 4806),
     ('make', 4722),
     ('plot', 4154),
     ('movies', 4080),
     ('acting', 4056),
     ('way', 3989),
     ('think', 3643)]



Aunque hayamos quitado las palabras vacías de un diccionario, hay un gran número de palabras vacías intrínsecas a nuestro dominio. En nuestro caso estas palabras que no aportan valor a la hora de distinguir entre una polarización positivia o negativa deberían ser consideradas como palabras vacías. Un ejemplo de estas palabras son muchas de las que aparecen en las listas de arriba. film, movie, acting y muchos nombres de actores y películas.

A continuación calculamos el ratio de las palabras positivas entre las negativas, esto nos indicará si una palabra es muy positiva, neutra o nada positiva. 

La forma de calcular este ratio de frecuencia es:
`número de usos positivos / (número de usos negativos+1)`

Se le añade `+1` al denominador para no dividir entre 0.


```python
MIN_FREQ = 200
positive_negative_prop = Counter()

for word,freq in list(total_freq.most_common()):
    if(freq > MIN_FREQ):
        proportion = positive_freq[word] / float(negative_freq[word]+1)
        positive_negative_prop[word] = proportion
```

Examinamos el ratio de algunas palabras:


```python
def check_words(words_list):
  for word_to_check in words_list:
    print("Word '%s' = %s"%(word_to_check, positive_negative_prop[word_to_check]))
check_words(['film', 'fantastic', 'bad'])
```

    Word 'film' = 1.089390707112753
    Word 'fantastic' = 4.503448275862069
    Word 'bad' = 0.2576330721426641


Como podemos ver, las palabras positivas tendrán valores muy altos. (>1)
Las palabras neutrales que aparecen en reviews positivas o negativas, tendrán valores muy cercanos a 1. (Equilibradas)
Y las palabras negativas estarán muy próximas a 0.


```python
for word,ratio in positive_negative_prop.most_common():
    positive_negative_prop[word] = np.log(ratio)
```

Una forma sencilla de normalizar estos valores y conseguir que las palabras neutrales estén en torno al 0 en vez de entorno al 1, es usando la función logaritmo.

A continuación comprobamos las mismas palabras anterioremente comprobadas y observamos los nuevos valores normalizados.


```python
check_words(['film', 'fantastic', 'bad'])
```

    Word 'film' = 0.08561855565085673
    Word 'fantastic' = 1.5048433868558566
    Word 'bad' = -1.3562189073456823


Arriba vemos que film, apenas aporta un valor discriminatorio.

A continuación vemos la lista de palabras más polarizadas y sus nuevos valores.


```python
positive_negative_prop.most_common(20)
```




    [('victoria', 2.681021528714291),
     ('captures', 2.038619547159581),
     ('wonderfully', 2.0218960560332353),
     ('powell', 1.978345424808467),
     ('refreshing', 1.8551812956655511),
     ('delightful', 1.8002701588959635),
     ('beautifully', 1.7626953362841438),
     ('underrated', 1.7197859696029656),
     ('superb', 1.7091514458966952),
     ('welles', 1.667706820558076),
     ('sinatra', 1.6389967146756448),
     ('touching', 1.637217476541176),
     ('stewart', 1.611998733295774),
     ('brilliantly', 1.5950491749820008),
     ('friendship', 1.5677652160335325),
     ('wonderful', 1.5645425925262093),
     ('magnificent', 1.54663701119507),
     ('finest', 1.546259010812569),
     ('jackie', 1.5439233053234738),
     ('freedom', 1.5091151908062312)]




```python
list(reversed(positive_negative_prop.most_common()))[0:20]
```




    [('unfunny', -2.6922395950755678),
     ('waste', -2.6193845640165536),
     ('pointless', -2.4553061800117097),
     ('redeeming', -2.3682390632154826),
     ('lousy', -2.307572634505085),
     ('worst', -2.286987896180378),
     ('laughable', -2.264363880173848),
     ('awful', -2.227194247027435),
     ('poorly', -2.2207550747464135),
     ('sucks', -1.987068221548821),
     ('lame', -1.981767458946166),
     ('insult', -1.978345424808467),
     ('horrible', -1.9102590939512902),
     ('amateurish', -1.9095425048844386),
     ('pathetic', -1.9003933102308506),
     ('wasted', -1.8382794848629478),
     ('crap', -1.8281271133989299),
     ('tedious', -1.802454758344803),
     ('dreadful', -1.7725281073001673),
     ('badly', -1.753626599532611)]



La aparición de "Victoria" parece indicar que sus películas tienen asociadas muy buenas críticas. Pero sabemos que en inglés hace referencia a un nombre propio, por lo que  para mejorar nuestra predicción habría considerar los nombres propios como stopwords.


```python
pd.DataFrame(positive_negative_prop.most_common()).plot.hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ffb79d86fd0>




![png](output_31_1.png)


Este histograma nos enseña la polaridad de las palabras en todo el corpus. Podemos observar que sigue una distribución normal con media en torno al 0. Es decir, la mayoría de las palabras están categorizadas como neutrales. Esto es ruido en nuestro clasificador. Esto se podría solucionar teniendo en cuenta aquellas palabas que aporten un valor discriminatorio mayor de |0.5|

# GENERANDO NUESTRO VOCABULARIO
A continuación para "tokenizar" nuestros textos y convertirlos en vector de palabras, vamos a crear un vocabulario que será la entrada de nuestra red neuronal.


```python
vocab = set(total_freq.keys())
```


```python
vocab_size = len(vocab)
print(vocab_size)
```

    73759



```python
word2index = {}
for i,word in enumerate(vocab):  
    word2index[word] = i
word2index
```




    {'': 0,
     'pork': 1,
     'cancer': 2,
     'hypermacho': 3,
     'beam': 4,
     'didja': 5,
     'sires': 6,
     'colonised': 7,
     'jest': 8,
     'fem': 9,
     'mitochondrial': 10,
     'azuma': 11,
     'stunk': 12,
     'attracting': 13,
     'cathernine': 14,
     'ventricle': 15,
     'ding': 16,
     'religous': 17,
     'training': 18,
     'cranks': 19,
     'hobbs': 20,
     'novac': 21,
     'millennia': 22,
     'zinn': 23,
     'sacrilage': 24,
     'mistry': 25,
     'sensualists': 26,
     'giff': 27,
     'bungling': 28,
     'raechel': 29,
     'swedes': 30,
     'miffed': 31,
     'ultimate': 32,
     'dought': 33,
     'plagiaristic': 34,
     'limned': 35,
     'jee': 36,
     'aracnophobia': 37,
     'centerpiece': 38,
     'unfaithal': 39,
     'knievel': 40,
     'ecstacy': 41,
     'trudged': 42,
     'alun': 43,
     'habituation': 44,
     'cannibalism': 45,
     'alarmist': 46,
     'looney': 47,
     'sudser': 48,
     'min': 49,
     'michelle': 50,
     'winninger': 51,
     'deployment': 52,
     'menzel': 53,
     'demonstrative': 54,
     'overpowered': 55,
     'seema': 56,
     'psychotics': 57,
     'coughthe': 58,
     'rollin': 59,
     'interferring': 60,
     'shimbei': 61,
     'orientated': 62,
     'traumatized': 63,
     'meriwether': 64,
     'kind': 65,
     'gruff': 66,
     'palsey': 67,
     'substories': 68,
     'acquittal': 69,
     'movecheck': 70,
     'compromised': 71,
     'zarustica': 72,
     'maadri': 73,
     'kaiser': 74,
     'budgetary': 75,
     'mt': 76,
     'factors': 77,
     'goulding': 78,
     'transposing': 79,
     'chineese': 80,
     'herbal': 81,
     'orkly': 82,
     'murderer': 83,
     'stephan': 84,
     'tage': 85,
     'forefathers': 86,
     'plays': 87,
     'dysfunction': 88,
     'gramophone': 89,
     'pendleton': 90,
     'juxtapositions': 91,
     'upto': 92,
     'excitement': 93,
     'ruphert': 94,
     'ultimo': 95,
     'mallorquins': 96,
     'lunacy': 97,
     'pratfalls': 98,
     'skyraiders': 99,
     'varela': 100,
     'rexes': 101,
     'mattresses': 102,
     'shvollenpecker': 103,
     'oversexed': 104,
     'taiwanese': 105,
     'toyota': 106,
     'neds': 107,
     'sugarman': 108,
     'facebuster': 109,
     'doel': 110,
     'veal': 111,
     'druidic': 112,
     'wary': 113,
     'extravaganzas': 114,
     'spiteful': 115,
     'sublime': 116,
     'nyfd': 117,
     'enthuses': 118,
     'wheaton': 119,
     'pharmaceutical': 120,
     'fulfill': 121,
     'innocence': 122,
     'undertake': 123,
     'infantile': 124,
     'crapfest': 125,
     'nec': 126,
     'shroyer': 127,
     'flour': 128,
     'valseuses': 129,
     'text': 130,
     'breasted': 131,
     'tachigui': 132,
     'additives': 133,
     'vanlint': 134,
     'mcphillip': 135,
     'impersonated': 136,
     'fictionalization': 137,
     'hitler': 138,
     'burry': 139,
     'curses': 140,
     'worn': 141,
     'thirbly': 142,
     'spitted': 143,
     'calhoun': 144,
     'hoyden': 145,
     'peculiarities': 146,
     'crops': 147,
     'blinding': 148,
     'gossemar': 149,
     'genghis': 150,
     'dusting': 151,
     'mausoleum': 152,
     'braincell': 153,
     'carrer': 154,
     'thumper': 155,
     'wale': 156,
     'beresford': 157,
     'coleman': 158,
     'deix': 159,
     'porkys': 160,
     'weasel': 161,
     'norton': 162,
     'garmes': 163,
     'croquet': 164,
     'aristocats': 165,
     'cigliutti': 166,
     'amore': 167,
     'casket': 168,
     'pending': 169,
     'mutated': 170,
     'probate': 171,
     'favourable': 172,
     'grandeurs': 173,
     'cavelleri': 174,
     'exasperated': 175,
     'kak': 176,
     'conflictive': 177,
     'paradoxically': 178,
     'aamir': 179,
     'aauugghh': 180,
     'onhand': 181,
     'deshimaru': 182,
     'strolls': 183,
     'grete': 184,
     'sickroom': 185,
     'clouded': 186,
     'baguettes': 187,
     'unabsorbing': 188,
     'sarajevo': 189,
     'sulk': 190,
     'chart': 191,
     'explore': 192,
     'permitted': 193,
     'malkovichian': 194,
     'whys': 195,
     'schlitz': 196,
     'disingenuous': 197,
     'hustle': 198,
     'immortel': 199,
     'insightfully': 200,
     'workforces': 201,
     'lyndon': 202,
     'aden': 203,
     'dunham': 204,
     'disbelieving': 205,
     'dunbar': 206,
     'segal': 207,
     'laroche': 208,
     'shakespearian': 209,
     'peasant': 210,
     'retention': 211,
     'concerted': 212,
     'serve': 213,
     'getz': 214,
     'discos': 215,
     'fused': 216,
     'looong': 217,
     'deceiving': 218,
     'ancients': 219,
     'brigadier': 220,
     'sistahs': 221,
     'violin': 222,
     'unengineered': 223,
     'deranged': 224,
     'lachlin': 225,
     'veoh': 226,
     'clung': 227,
     'ran': 228,
     'swabby': 229,
     'rataud': 230,
     'endearment': 231,
     'comity': 232,
     'bookend': 233,
     'waaaaaayyyy': 234,
     'siren': 235,
     'misleads': 236,
     'alrite': 237,
     'examination': 238,
     'panned': 239,
     'themsleves': 240,
     'wandered': 241,
     'simper': 242,
     'pliers': 243,
     'rump': 244,
     'cripplingly': 245,
     'scrawl': 246,
     'lewinski': 247,
     'gearheads': 248,
     'ktla': 249,
     'ambience': 250,
     'dozens': 251,
     'presumes': 252,
     'awards': 253,
     'surpressors': 254,
     'edits': 255,
     'difficulties': 256,
     'remar': 257,
     'wheelchairs': 258,
     'fiascos': 259,
     'claimed': 260,
     'waldeman': 261,
     'dangles': 262,
     'aloud': 263,
     'luncheon': 264,
     'cliffhangers': 265,
     'reminding': 266,
     'protected': 267,
     'serafinowicz': 268,
     'sorrell': 269,
     'bused': 270,
     'vulnerability': 271,
     'kaleidoscope': 272,
     'lizard': 273,
     'plateful': 274,
     'subbed': 275,
     'mpkdh': 276,
     'majkowski': 277,
     'eroticism': 278,
     'latecomers': 279,
     'outreach': 280,
     'visualizes': 281,
     'ramotswe': 282,
     'scientific': 283,
     'mcgaw': 284,
     'zb': 285,
     'mole': 286,
     'macho': 287,
     'uninstructive': 288,
     'resourceful': 289,
     'pumba': 290,
     'soleil': 291,
     'whopper': 292,
     'adhering': 293,
     'slobber': 294,
     'ai': 295,
     'lifelike': 296,
     'finisher': 297,
     'eponymous': 298,
     'shoudln': 299,
     'oyl': 300,
     'carrefour': 301,
     'argonne': 302,
     'golovanov': 303,
     'gunmen': 304,
     'palestinians': 305,
     'precocious': 306,
     'teapot': 307,
     'somtimes': 308,
     'aiden': 309,
     'curmudgeon': 310,
     'opting': 311,
     'imagery': 312,
     'stitches': 313,
     'irresistibly': 314,
     'ezra': 315,
     'hypesters': 316,
     'spritely': 317,
     'honeymooners': 318,
     'mined': 319,
     'muggings': 320,
     'fallow': 321,
     'grimm': 322,
     'fiddler': 323,
     'daneille': 324,
     'carelessness': 325,
     'braveheart': 326,
     'cahoots': 327,
     'reflexivity': 328,
     'agekudos': 329,
     'abdu': 330,
     'tick': 331,
     'kindling': 332,
     'flowed': 333,
     'terrifically': 334,
     'montegna': 335,
     'rest': 336,
     'unperceptive': 337,
     'fannin': 338,
     'hindersome': 339,
     'monique': 340,
     'einstein': 341,
     'lea': 342,
     'portrayed': 343,
     'garrett': 344,
     'arcaica': 345,
     'parlor': 346,
     'blight': 347,
     'abusing': 348,
     'gainful': 349,
     'infects': 350,
     'twiggy': 351,
     'storszek': 352,
     'tediousness': 353,
     'tigerland': 354,
     'spirited': 355,
     'skipping': 356,
     'gills': 357,
     'barrels': 358,
     'soni': 359,
     'guanajuato': 360,
     'burkhalter': 361,
     'ingela': 362,
     'emulations': 363,
     'estefan': 364,
     'adlai': 365,
     'trainor': 366,
     'attraction': 367,
     'adma': 368,
     'flippantly': 369,
     'irritated': 370,
     'pendant': 371,
     'annoyed': 372,
     'storaro': 373,
     'az': 374,
     'punters': 375,
     'radical': 376,
     'unresponsive': 377,
     'printer': 378,
     'hmmmmmmmm': 379,
     'portrayer': 380,
     'gained': 381,
     'lars': 382,
     'willed': 383,
     'appreciation': 384,
     'herilhy': 385,
     'campy': 386,
     'fahrenheit': 387,
     'rodrix': 388,
     'nordham': 389,
     'underfoot': 390,
     'woolgathering': 391,
     'bs': 392,
     'aldonova': 393,
     'elequence': 394,
     'suspending': 395,
     'incubates': 396,
     'sans': 397,
     'misfire': 398,
     'reassuring': 399,
     'jerri': 400,
     'rework': 401,
     'utilities': 402,
     'handlers': 403,
     'margineanus': 404,
     'cos': 405,
     'masters': 406,
     'widened': 407,
     'excuse': 408,
     'pinkish': 409,
     'split': 410,
     'kewl': 411,
     'attract': 412,
     'wavy': 413,
     'alda': 414,
     'recognizable': 415,
     'whip': 416,
     'securing': 417,
     'insular': 418,
     'idiosyncratic': 419,
     'hayseed': 420,
     'tukur': 421,
     'advisedly': 422,
     'proposal': 423,
     'espeically': 424,
     'astrotech': 425,
     'shoufukutei': 426,
     'muncie': 427,
     'notoriety': 428,
     'escapism': 429,
     'outburst': 430,
     'hipper': 431,
     'condon': 432,
     'prix': 433,
     'glop': 434,
     'lespart': 435,
     'occupational': 436,
     'slacken': 437,
     'kerkhof': 438,
     'gymnasts': 439,
     'rigorous': 440,
     'jame': 441,
     'definetly': 442,
     'someway': 443,
     'caresses': 444,
     'deepak': 445,
     'sutdying': 446,
     'da': 447,
     'groundwork': 448,
     'ford': 449,
     'pentimento': 450,
     'hanns': 451,
     'drab': 452,
     'der': 453,
     'underwear': 454,
     'casper': 455,
     'puppetry': 456,
     'pakis': 457,
     'pearlman': 458,
     'bets': 459,
     'deservingly': 460,
     'hesitates': 461,
     'liberty': 462,
     'inconvenience': 463,
     'grosbard': 464,
     'steam': 465,
     'mounts': 466,
     'warnercolor': 467,
     'matt': 468,
     'beatific': 469,
     'colwell': 470,
     'slumping': 471,
     'doings': 472,
     'miswrote': 473,
     'jodoworsky': 474,
     'floods': 475,
     'enticement': 476,
     'rigueur': 477,
     'starsky': 478,
     'nick': 479,
     'monumentous': 480,
     'naffness': 481,
     'scratched': 482,
     'mays': 483,
     'starblazers': 484,
     'doves': 485,
     'wellpaced': 486,
     'growls': 487,
     'mist': 488,
     'ropes': 489,
     'baltimoreans': 490,
     'touch': 491,
     'aja': 492,
     'valga': 493,
     'recur': 494,
     'contreras': 495,
     'unbearded': 496,
     'cassetti': 497,
     'cascading': 498,
     'megapack': 499,
     'bandido': 500,
     'sprays': 501,
     'smuttiness': 502,
     'ladder': 503,
     'dosage': 504,
     'milwall': 505,
     'competent': 506,
     'hilltop': 507,
     'discomfort': 508,
     'stutter': 509,
     'draughtswoman': 510,
     'stockpile': 511,
     'littlekuriboh': 512,
     'bootie': 513,
     'disappoints': 514,
     'koz': 515,
     'proceeded': 516,
     'solimeno': 517,
     'avian': 518,
     'wicked': 519,
     'scales': 520,
     'howls': 521,
     'pleasaunces': 522,
     'shead': 523,
     'wickerman': 524,
     'xylophonist': 525,
     'companys': 526,
     'lorado': 527,
     'undertook': 528,
     'utopia': 529,
     'chihiro': 530,
     'courtesan': 531,
     'democratically': 532,
     'broad': 533,
     'conniving': 534,
     'photographic': 535,
     'davidbathsheba': 536,
     'glum': 537,
     'militaries': 538,
     'unfairly': 539,
     'ohio': 540,
     'talosian': 541,
     'grafted': 542,
     'cof': 543,
     'evers': 544,
     'bogglingly': 545,
     'overheating': 546,
     'mammothly': 547,
     'unfurnished': 548,
     'loves': 549,
     'battle': 550,
     'qi': 551,
     'tragedy': 552,
     'blonde': 553,
     'dystopic': 554,
     'cineasts': 555,
     'antonius': 556,
     'tarka': 557,
     'bloodthirst': 558,
     'milieu': 559,
     'vivant': 560,
     'censured': 561,
     'stinkpile': 562,
     'differential': 563,
     'affirmation': 564,
     'lydia': 565,
     'superlivemation': 566,
     'financially': 567,
     'pac': 568,
     'funiest': 569,
     'revolving': 570,
     'applauds': 571,
     'sperr': 572,
     'sybil': 573,
     'pedestrians': 574,
     'promise': 575,
     'elam': 576,
     'gazongas': 577,
     'categorised': 578,
     'tura': 579,
     'jeb': 580,
     'opportune': 581,
     'furgusson': 582,
     'irl': 583,
     'refuge': 584,
     'enacting': 585,
     'disenchantment': 586,
     'tis': 587,
     'breads': 588,
     'transposed': 589,
     'sivan': 590,
     'johan': 591,
     'siu': 592,
     'beswick': 593,
     'vlkava': 594,
     'auburn': 595,
     'gurl': 596,
     'figuring': 597,
     'numbingly': 598,
     'soft': 599,
     'centred': 600,
     'harrowed': 601,
     'hearkens': 602,
     'joeseph': 603,
     'moovies': 604,
     'witchie': 605,
     'cigs': 606,
     'stage': 607,
     'mitevska': 608,
     'roulette': 609,
     'rolly': 610,
     'ramchand': 611,
     'mulit': 612,
     'ameteurish': 613,
     'supplicant': 614,
     'compositor': 615,
     'pointer': 616,
     'dooooosie': 617,
     'rembrandt': 618,
     'skolimowski': 619,
     'vangelis': 620,
     'dzundza': 621,
     'cherri': 622,
     'harvested': 623,
     'filmmakers': 624,
     'essendon': 625,
     'nicolie': 626,
     'reassigned': 627,
     'calvins': 628,
     'refinery': 629,
     'amrish': 630,
     'lesson': 631,
     'nris': 632,
     'clerical': 633,
     'oooo': 634,
     'medication': 635,
     'phenomenons': 636,
     'santoni': 637,
     'moronfest': 638,
     'soviet': 639,
     'harden': 640,
     'relationsip': 641,
     'roofer': 642,
     'afar': 643,
     'neptune': 644,
     'unforgetable': 645,
     'sorcha': 646,
     'ditz': 647,
     'mehemet': 648,
     'advice': 649,
     'romantisised': 650,
     'ulcerating': 651,
     'millimeter': 652,
     'snorer': 653,
     'glady': 654,
     'daylights': 655,
     'anorexia': 656,
     'gettysburg': 657,
     'foe': 658,
     'suck': 659,
     'ising': 660,
     'johar': 661,
     'cradled': 662,
     'womennone': 663,
     'clampets': 664,
     'ishwar': 665,
     'dandies': 666,
     'jughead': 667,
     'themself': 668,
     'chundering': 669,
     'shipment': 670,
     'owed': 671,
     'wrestlemanias': 672,
     'commercisliation': 673,
     'vooren': 674,
     'shipped': 675,
     'brogues': 676,
     'nectar': 677,
     'kitties': 678,
     'buyer': 679,
     'tapers': 680,
     'leidner': 681,
     'perverted': 682,
     'vaticani': 683,
     'insouciance': 684,
     'iannaccone': 685,
     'succulently': 686,
     'apprehending': 687,
     'mitchel': 688,
     'workday': 689,
     'titty': 690,
     'oppenheimer': 691,
     'eser': 692,
     'tassel': 693,
     'sumptuousness': 694,
     'intonations': 695,
     'cherubic': 696,
     'franklin': 697,
     'propane': 698,
     'senegalese': 699,
     'compiled': 700,
     'arret': 701,
     'intrusively': 702,
     'wrinkle': 703,
     'urmila': 704,
     'buds': 705,
     'librarians': 706,
     'cubbyholes': 707,
     'portends': 708,
     'interconnecting': 709,
     'posterity': 710,
     'norseman': 711,
     'episodic': 712,
     'bleating': 713,
     'frumpy': 714,
     'ofcourse': 715,
     'rouged': 716,
     'voerhoven': 717,
     'stun': 718,
     'beret': 719,
     'scrutinized': 720,
     'sequenes': 721,
     'inhumanity': 722,
     'merkle': 723,
     'vomitum': 724,
     'gobbler': 725,
     'plastique': 726,
     'frownbuster': 727,
     'turaqui': 728,
     'sanju': 729,
     'x': 730,
     'chakraborty': 731,
     'curator': 732,
     'strategies': 733,
     'orientals': 734,
     'poorly': 735,
     'glass': 736,
     'fellowship': 737,
     'spaz': 738,
     'decomp': 739,
     'warbler': 740,
     'aonghas': 741,
     'withouts': 742,
     'bergqvist': 743,
     'dutt': 744,
     'maclaine': 745,
     'prowls': 746,
     'millie': 747,
     'turbulent': 748,
     'clunks': 749,
     'shards': 750,
     'conaughey': 751,
     'pounced': 752,
     'lineal': 753,
     'justicia': 754,
     'ksm': 755,
     'parnell': 756,
     'alcoholic': 757,
     'seafood': 758,
     'marienbad': 759,
     'mander': 760,
     'rowdy': 761,
     'designates': 762,
     'cheerless': 763,
     'hallgren': 764,
     'bastidge': 765,
     'aubrey': 766,
     'panoramas': 767,
     'ke': 768,
     'blige': 769,
     'nicks': 770,
     'taunts': 771,
     'thingie': 772,
     'zerifferelli': 773,
     'fisticuff': 774,
     'dakota': 775,
     'stettner': 776,
     'relaxers': 777,
     'cared': 778,
     'entrenchments': 779,
     'jaipur': 780,
     'rosco': 781,
     'murkily': 782,
     'karogi': 783,
     'sharpe': 784,
     'msb': 785,
     'kelemen': 786,
     'anal': 787,
     'entities': 788,
     'hagerthy': 789,
     'hyderabadi': 790,
     'indulgent': 791,
     'chicatillo': 792,
     'capabilities': 793,
     'leguizamo': 794,
     'couleur': 795,
     'apostrophe': 796,
     'uncynical': 797,
     'sadomasochism': 798,
     'retreated': 799,
     'kimmell': 800,
     'artistry': 801,
     'helen': 802,
     'stagnation': 803,
     'globalizing': 804,
     'puh': 805,
     'prosaically': 806,
     'redhead': 807,
     'footsteps': 808,
     'longtime': 809,
     'axiomatic': 810,
     'fans': 811,
     'xtianity': 812,
     'alucard': 813,
     'predominant': 814,
     'lynchings': 815,
     'fielding': 816,
     'contessa': 817,
     'fried': 818,
     'abortive': 819,
     'underscored': 820,
     'adroitly': 821,
     'awkwardness': 822,
     'sinese': 823,
     'travelcard': 824,
     'maryam': 825,
     'intact': 826,
     'ads': 827,
     'northam': 828,
     'nafta': 829,
     'matlock': 830,
     'madchen': 831,
     'swung': 832,
     'numero': 833,
     'genetics': 834,
     'ashley': 835,
     'scot': 836,
     'zeffirelli': 837,
     'slayers': 838,
     'duquenne': 839,
     'quibble': 840,
     'fumes': 841,
     'zues': 842,
     'pap': 843,
     'lasciviousness': 844,
     'cukor': 845,
     'lemuria': 846,
     'pejorative': 847,
     'toto': 848,
     'midway': 849,
     'vadis': 850,
     'sliced': 851,
     'businesspeople': 852,
     'homey': 853,
     'artisticly': 854,
     'refracted': 855,
     'dysfunctions': 856,
     'atlanteans': 857,
     'baby': 858,
     'bassis': 859,
     'reconstructions': 860,
     'johannesburg': 861,
     'jaret': 862,
     'hungarian': 863,
     'useless': 864,
     'indestructible': 865,
     'jacuzzi': 866,
     'kayyyy': 867,
     'mi': 868,
     'milinkovic': 869,
     'dioz': 870,
     'discombobulation': 871,
     'abm': 872,
     'vise': 873,
     'lovesick': 874,
     'faraway': 875,
     'unheated': 876,
     'bogie': 877,
     'messmer': 878,
     'foreseeing': 879,
     'labouf': 880,
     'phoenicia': 881,
     'overhears': 882,
     'remunda': 883,
     'unraveling': 884,
     'daisies': 885,
     'aristide': 886,
     'bedingfield': 887,
     'cheesecake': 888,
     'terrace': 889,
     'flagship': 890,
     'pickup': 891,
     'escalating': 892,
     'uttara': 893,
     'gunshots': 894,
     'meres': 895,
     'savales': 896,
     'horrorfilm': 897,
     'sheeple': 898,
     'twine': 899,
     'aboriginies': 900,
     'hoydenish': 901,
     'reshipping': 902,
     'wouln': 903,
     'speckle': 904,
     'befuddled': 905,
     'liebe': 906,
     'mopey': 907,
     'steffen': 908,
     'noises': 909,
     'install': 910,
     'barren': 911,
     'workaholics': 912,
     'alcoholism': 913,
     'bashing': 914,
     'ilu': 915,
     'disrupts': 916,
     'republics': 917,
     'briliant': 918,
     'rheubottom': 919,
     'eludes': 920,
     'endearing': 921,
     'alexanderplatz': 922,
     'judgment': 923,
     'colorfully': 924,
     'darlene': 925,
     'buckaroo': 926,
     'machettes': 927,
     'moncia': 928,
     'gaita': 929,
     'doctresses': 930,
     'thunderbolt': 931,
     'flak': 932,
     'vanquishes': 933,
     'supermen': 934,
     'shuddup': 935,
     'pinkie': 936,
     'sensations': 937,
     'elvira': 938,
     'guessed': 939,
     'consults': 940,
     'amatuerish': 941,
     'corsair': 942,
     'munitions': 943,
     'git': 944,
     'hectic': 945,
     'septic': 946,
     'flapper': 947,
     'atherton': 948,
     'anons': 949,
     'motos': 950,
     'bambou': 951,
     'childish': 952,
     'reviczky': 953,
     'graystone': 954,
     'mandate': 955,
     'heightens': 956,
     'petron': 957,
     'rods': 958,
     'excell': 959,
     'collection': 960,
     'macrae': 961,
     'wiping': 962,
     'delli': 963,
     'poltergeist': 964,
     'minutia': 965,
     'malikka': 966,
     'upbringings': 967,
     'noisier': 968,
     'ifyou': 969,
     'manchu': 970,
     'prophets': 971,
     'mice': 972,
     'leit': 973,
     'morrocco': 974,
     'korman': 975,
     'reviving': 976,
     'slowed': 977,
     'epstein': 978,
     'testified': 979,
     'bassett': 980,
     'bendan': 981,
     'punkris': 982,
     'scolded': 983,
     'okinawan': 984,
     'poisoning': 985,
     'blueprints': 986,
     'impalement': 987,
     'bethune': 988,
     'ted': 989,
     'dissertations': 990,
     'oops': 991,
     'deesh': 992,
     'chaya': 993,
     'gunnerside': 994,
     'mano': 995,
     'advision': 996,
     'negro': 997,
     'postino': 998,
     'dumbed': 999,
     ...}



# CLASIFICADOR MEDIANTE RED NEURONAL

Ya nos acercamos al final de nuestro analizador de sentimientos. Hemos decidido utilizar una red neuronal clásica, que como ya hemos comentado en la introducción posee 3 capas. La primera que es de entradas y tiene la longitud de nuestro vocabulario. Cada review se codificará como un vector en el que cada cada elemento representa la frequencia de apariciones de esa palabra en el texto. La capa intermedia tiene 30 nodos, y finalmente la capa final tiene un solo nodo de salida.

Nuestro clasificador puede en modo entrenamiento y test devuelve 2 etiquetas Positivo o Negativo. Y en modo interactivo (logger), además devuelve el nivel de confianza y una tercera etiqueta Neutral.


```python
class SentimentReviewClassifier:
    def __init__(self, learning_rate = 0.01):
        np.random.seed(7)
        self.input_nodes = len(vocab)
        self.middle_nodes = 30
        self.final_nodes = 1
        self.learning_rate = learning_rate

        # Initialize net
        self.hidden_0_1 = np.zeros((self.input_nodes,self.middle_nodes))
        self.hidden_1_2 = np.random.normal(0.0, self.middle_nodes**-0.5, 
                                                (self.middle_nodes, self.final_nodes))
        self.first_layer = np.zeros((1,self.middle_nodes))
    
    translate_label = {
        'POSITIVE': 1,
        'NEGATIVE': 0,
    }
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    def show_progress(self, i, total, correct):
        progress =  str(100 * i/float(total))[:4]
        accuracy =  str(correct * 100 / float(i+1))[:4]
        sys.stdout.write("\r - Progress:%s %%| Correct:%s | Accuracy:%s%%"%(progress, correct, accuracy))
        
    def train(self, reviews_corpus, labels_t):
        """ Update weights from corpus"""
        reviews = list()
        for review in reviews_corpus:
            indices = set()
            for word in remove_stopwords(review):
                if(word in word2index.keys()):
                    indices.add(word2index[word])
            reviews.append(list(indices))

        correct = 0
        for i in range(len(reviews)):
            review = reviews[i]
            label = labels_t[i]
            # Training
            self.first_layer *= 0
            for index in review:
                self.first_layer += self.hidden_0_1[index]
            second_layer = self.sigmoid(self.first_layer.dot(self.hidden_1_2))            

            # Output error
            second_layer_error = second_layer - self.translate_label[label]
            second_layer_delta = second_layer_error * self.sigmoid_output_2_derivative(second_layer)

            # Backpropagated error
            first_layer_error = second_layer_delta.dot(self.hidden_1_2.T)
            first_layer_delta = first_layer_error
            self.hidden_1_2 -= self.first_layer.T.dot(second_layer_delta) * self.learning_rate

            for index in review:
                self.hidden_0_1[index] -= first_layer_delta[0] * self.learning_rate

            if(second_layer >= 0.5 and label == 'POSITIVE'):
                correct += 1
            elif(second_layer < 0.5 and label == 'NEGATIVE'):
                correct += 1
            self.show_progress(i, len(reviews), correct)
            if(i % 2500 == 0):
                print("")
    
    def test(self, reviews, testing_labels):
        """ Test and don't update the weights """
        correct = 0
        for i in range(len(reviews)):
            pred = self.run(reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            self.show_progress(i, len(reviews), correct)
    
    def run(self, review, logger = False):
        """ Evaluate a single review"""
        self.first_layer *= 0
        unique_indices = set()
        for word in remove_stopwords(review):
            if word in word2index.keys():
                unique_indices.add(word2index[word])
        for index in unique_indices:
            self.first_layer += self.hidden_0_1[index]
        second_layer = self.sigmoid(self.first_layer.dot(self.hidden_1_2))
        out = second_layer[0]
        threshold = 0
        if logger:
          print(out)
          threshold = 0.05
        if out >= 0.5 + threshold:
            return "POSITIVE"
        elif out < 0.5 - threshold:
            return "NEGATIVE"
        else:
            return "NEUTRAL"

```


```python
# Dividimos el dataset en 70% Training y 30% Testing
DROPOUT_PARTITION = 0.7
SPLIT_PART = int(len(df)*DROPOUT_PARTITION)
df_train = df.iloc[:SPLIT_PART]
df_test = df.iloc[SPLIT_PART:]
print("DROPOUT SPLIT Train: %d, Test: %d, TOTAL: %d"%(len(df_train), len(df_test), len(df_train)+len(df_test)))
df_test.head()
```

    DROPOUT SPLIT Train: 17500, Test: 7500, TOTAL: 25000





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>labels</th>
      <th>reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17500</th>
      <td>POSITIVE</td>
      <td>one reason pixar has endured so well  and been...</td>
    </tr>
    <tr>
      <th>17501</th>
      <td>NEGATIVE</td>
      <td>i saw the film and i got screwed  because the ...</td>
    </tr>
    <tr>
      <th>17502</th>
      <td>POSITIVE</td>
      <td>a scanner darkly  minority report  blade runne...</td>
    </tr>
    <tr>
      <th>17503</th>
      <td>NEGATIVE</td>
      <td>what  s happening to rgv  he seems to repeat h...</td>
    </tr>
    <tr>
      <th>17504</th>
      <td>POSITIVE</td>
      <td>i  ve seen this film in avant  premiere at ima...</td>
    </tr>
  </tbody>
</table>
</div>




```python
net = SentimentReviewClassifier(learning_rate=0.02)
net.train(df_train['reviews'],df_train['labels'])
```

     - Progress:0.0 %| Correct:1 | Accuracy:100.%
     - Progress:14.2 %| Correct:2028 | Accuracy:81.0%
     - Progress:28.5 %| Correct:4129 | Accuracy:82.5%
     - Progress:42.8 %| Correct:6277 | Accuracy:83.6%
     - Progress:57.1 %| Correct:8469 | Accuracy:84.6%
     - Progress:71.4 %| Correct:10629 | Accuracy:85.0%
     - Progress:85.7 %| Correct:12785 | Accuracy:85.2%
     - Progress:99.9 %| Correct:14934 | Accuracy:85.3%


```python
# Cambiamos la forma de indexar por problemas en algún dato en el dataframe.
net.test(df_test.iloc[:,1].values, df_test.iloc[:,0].values)
```

     - Progress:99.9 %| Correct:6453 | Accuracy:86.0%


```python
# Example of a single review
net.run('This a great film fantastic actors', logger=True)
```

    [0.69505086]





    'POSITIVE'




```python
#@title ### Inserta un texto para probar el analizador de sentimientos
review = "This movie is the best in the world" #@param {type:"string"}
```


```python
print('Review a analizar: "%s"'%review)
print('La Review es: %s'% net.run(review, logger=True))
```

    Review a analizar: "This movie is the best in the world"
    [0.6628306]
    La Review es: POSITIVE


# CONCLUSIONES






Como hemos podido observar, hemos obtenido resultados muy semejantes a los propuestos en el paper [1].

Existen multitud de aproximaciones a un mismo problema.

Determinar la polaridad positiva o negativa de una reseña se puede conseguir con modelos relativamente sencillos.

Obtener datos más precisos, como qué tipo de sentimiento expresa, enfado, ira, amor, felicidad son un reto todavía en investigación.

Técnicas muy similares propuestas en esta práctica se pueden utilizar para detectar reseñas fraudulentas, la dificultad está en conseguir un dataset etiquetado.

Un analizador de sentimiento se puede simplificar a un clasificador de textos, en el que cada tópico es el sentimiento que queremos clasificar.

Posibles mejoras, un mayor tratamiento en la reducción ruido, mediante la eliminación de palabras vacías aumentaría la precisión de nuestra red.

Existen algoritmos más avanzados que posilemente den mejores resultados. Modelizar el corpus como word embeddings es una alternativa. Otra opción sería utilizar LSTM (Long short-term memory) [2]. Ambos sistemas tienen en cuenta las palabras que están cercanas y tenemos seguridad de que producirían mejores resultados.

Aunque nosotros hemos tomado una vía muy rudimentaria para ir comentando y describiendo la metodología paso a paso, existen varias librerías que pueden simplificar nuestro código. Nosotros aconsejamos la utilización de estas librerías en sistemas reales. Algunas de estas librerías son: sklearn, pytorch, tensorflow, keras, nltk, scipy entre otras.

Por último quremos destacar que este analizador de sentimientos funcionará bien con el dominio de películas en el idioma inglés, pero no sería el más adecuado para corpus de otros dominios, y por supuesto el resultado no sería fiable en el caso de clasificar documentos que no tengan ninguna palabra de nuestro vocabulario.

De hecho al realizar esta prueba, se puede observar que el clasificador tiene un bias positio. Si no introduces ninguna palabra clasifica el texto como positivo con una confianza de 0.55.
