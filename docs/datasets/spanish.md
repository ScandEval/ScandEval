# 🇪🇸 Spanish

This is an overview of all the datasets used in the Spanish part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Sentiment headlines
This dataset was published in [this paper](https://arxiv.org/abs/2208.13947) and features political news headlines.

The original full dataset consists of 1,371 /  609 / 459 samples for training, validation, and testing, respectively. We use 861/  256 / 1024 samples for training, validation, and testing, respectively. All our splits are subsets of the original ones. The label distribution for the splits are as follows:

| Split | positive | negative | neutral | Total |
|-------|----------|----------|---------|--------|
| Train | 368      | 248      | 245     | 861    |
| Val   | 88       | 90       | 78      | 256    |
| Test  | 417      | 293      | 314     | 1,024  |
| Total | 873      | 631      | 637     | 2,141  |

Here are a few examples from the training split:

```json
{
    "text": "Mauricio Macri, en el cierre de campaña: “Esta marcha no termina hoy acá, sino en noviembre”",
    "label": "neutral"
}
```
```json
{
    "text": "Lavagna reforzó su discurso económico y pidió más consumo", "label": "positive"
}
```
```json
{
    "text": "Sin la aprobación del Fondo, Macri quema reservas para la fuga",
    "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Aquí hay textos y su sentimiento, que puede ser 'positivo' o 'negativo'.
  ```
- Base prompt template:
  ```
  Texto: {text}
  Sentimiento: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Texto: {text}

  Clasifica el sentimiento del texto. Responde con 'positivo' o 'negativo'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset sentiment-headlines-es
```


## Named Entity Recognition

## Linguistic Acceptability

## Reading Comprehension

### XQuAD-es

This dataset was published in [this paper](https://arxiv.org/abs/1910.11856) and contains 1190 question-answer pairs from [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) translated into ten languages by professional translators.

The dataset is split intro 550 / 128 / 512 question-answer pairs for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
    "context": "El Mercado del Grainger reemplazó a un mercado anterior construido originalmente en 1808 llamado el Mercado del Carnicero. El Mercado del Grainger en sí mismo, se abrió en 1835 y fue el primer mercado interior de Newcastle. En el momento de su apertura en 1835 se dijo que era uno de los mercados más grandes y hermosos de Europa. La inauguración se celebró con una gran cena a la que asistieron 2000 invitados, y la Galería de Arte Laing tiene un cuadro de este evento. Con la excepción del techo de madera, que fue destruido por un incendio en 1901 y sustituido por arcos de celosía de acero, el mercado se encuentra en su mayor parte en su estado original. La arquitectura del Mercado del Grainger, como la mayoría de las de Grainger Town, que están clasificadas en el grado I o II, fue clasificada en el grado I en 1954 por Patrimonio Inglés.",
    "question": "¿Cuántos invitados asistieron a la cena de inauguración del Mercado del Grainger?",
    "answer": {"answer_start": [396], "text": ["2000"]}
}
```
```json
{
    "context": "Los avances realizados en Oriente Medio en botánica y química llevaron a la medicina en el Islam medieval a desarrollar sustancialmente la farmacología. Muhammad ibn Zakarīya Rāzi (Rhazes) (865-915), por ejemplo, actuó para promover los usos médicos de los compuestos químicos. Abu al-Qasim al-Zahrawi (Abulcasis) (936-1013) fue pionero en la preparación de medicamentos por sublimación y destilación. Su Liber servitoris es de particular interés, ya que proporciona al lector recetas y explica cómo preparar los 'simples' a partir de los cuales se componían los complejos medicamentos que se utilizaban entonces de forma generalizada. Sabur Ibn Sahl (d 869), fue, sin embargo, el primer médico en iniciar la farmacopedia, describiendo una gran variedad de medicamentos y remedios para las dolencias. Al-Biruni (973-1050) escribió una de las obras islámicas más valiosas sobre farmacología, titulada Kitab al-Saydalah (El libro de los medicamentos), en la que detallaba las propiedades de los medicamentos y esbozaba el papel de la farmacia, así como las atribuciones y los deberes de los farmacéuticos. Avicena también describió nada menos que 700 preparados, sus propiedades, modos de acción y sus indicaciones. De hecho, dedicó todo un volumen a los medicamentos simples en El canon de la medicina. De gran impacto fueron también las obras de al-Maridini de Bagdad y El Cairo, y de Ibn al-Wafid (1008-1074), ambas impresas en latín más de cincuenta veces, apareciendo como De Medicinis universalibus et particularibus de 'Mesue' el más joven, y el Medicamentis simplicibus de 'Abenguefit'. Pedro de Abano (1250-1316) tradujo y añadió un suplemento a la obra de al-Maridini bajo el título De Veneris. Las contribuciones de Al-Muwaffaq en este campo también son pioneras. En su vida en el siglo X, escribió Los fundamentos de las verdaderas propiedades de los remedios, describiendo, entre otras cosas, el óxido arsenioso, y conociendo el ácido silícico. Hizo una clara distinción entre carbonato de sodio y carbonato de potasio y llamó la atención sobre la naturaleza venenosa de los compuestos de cobre, especialmente el vitriolo de cobre, y también los compuestos de plomo. También describe la destilación de agua de mar para beber [se requiere verificación].",
    "question": "¿Cuáles fueron los desarrollos en los que los científicos influyeron en la creación de la farmacología en el Islam medieval?",
    "answer": {"answer_start": [43], "text": ["botánica y química"]}
}
```
```json
{
    "id": "5725c91e38643c19005acced",
    "context": "A pesar de sus cuerpos blandos y gelatinosos, los fósiles que se cree que representan a los ctenóforos, aparentemente sin tentáculos pero con muchas más filas de púas que las formas modernas, han sido encontrados en lagerstätten en los primeros tiempos de la época de la era Cámbrica, hace alrededor de 515 millones de años. La posición de los ctenóforos en el árbol genealógico evolutivo de los animales se ha discutido durante mucho tiempo, y la opinión mayoritaria en la actualidad, basada en la filogenética molecular, es que los cnidarios y los bilaterianos están más estrechamente relacionados entre sí que cualquiera de ellos con los ctenóforos. Un análisis reciente de filogenética molecular concluyó que el antepasado común de todos los ctenóforos modernos era similar a los cidípidos, y que todos los grupos modernos aparecieron relativamente recientemente, probablemente después del evento de extinción del Cretácico-Paleógeno hace 66 millones de años. Las pruebas acumuladas desde la década de 1980 indican que los "cidípidos" no son monofiléticos, es decir, no incluyen a todos y solo a los descendientes de un único antepasado común, ya que todos los demás grupos tradicionales de ctenóforos son descendientes de varios cidípidos.",
    "question": "¿Qué edad tienen los fósiles encontrados que representan los ctenóforos?",
    "answer": {"answer_start": [303], "text": ["515 millones de años"]}
}
```

## Knowledge

## Common-sense Reasoning

## Summarization
