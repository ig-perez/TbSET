# What is this?
Este repositorio contiene un traductor Español-Inglés basado en un Transformer model entrenado con XXX ejemplos (customized training loop) en Google Cloud.

- [ ] IMAGEN DE UN TRANSFORMER

No se trata de un notebook, sino de un proyecto modular orientado a objetos con un TUI sencillo que permite interactuar con el traductor, para entrenarlo y también para hacer inferencia con el modelo guardado en disco después del entrenamiento.

- [ ] GIF EJEMPLO

Este no es proyecto para uso comercial, sino como un ejercicio práctico para incrementar mis conocimientos y habilidades técnicas en NLP. Por este motivo, en caso lo deseen utilizar, tengan en cuenta que está lejos de funcionar perfectamente. En definitiva, hay varias mejoras futuras, las cuales están detalladas en la sección "Future improvements". 

Puedes leer el artículo relacionado con este repositorio aquí. En el artículo detallo los retos más importantes a los que me enfrenté, profundizo en conceptos cruciales, y comparto algunos consejos prácticos para aquellos que estén en el mismo barco que yo.

Espero lo disfrutes!

## The project structure
- [ ] Add mermaid

## The configuration file
- [ ] Explain each section and element

## The dataset
- [ ] Explain OPUS/tf-datasets and its limitations

## The hardware used for training
- [ ] VM description, time, epochs, examples, limitations

## The development process
- [ ] Talk about IDE to remote GCP instance, reference to article on personal site.

# How can I use it?

## Requirements
xxx

## Steps to run the module
xxx

## Results
xxx

## Future improvements
- [ ] better dataset
- [ ] use test and validation sets

# Where can I learn more about NLP?


# Notes
- `local.multivac` is just a storage folder with a random name.

# General


1. Load dataset for training
2. Tokenize the datasets for training
    2.1. Create a vocabulary for our dataset using the Wordpiece algorithm
    2.2. Use the vocabulary to build a custom tokenizer based on BERT tokenization
    2.3. Create the tokenizers (objects) for both languages
3. Prepare input data for training (one line)
    3.1. Cache the datasets
    3.2. Shuffle the datasets
    3.3. Create batches
    3.4. Tokenize the datasets
    3.5. Prefetch the datasets

4. Train
    4.1. Instantiate the Transformer
    4.2. Set up the checkpoint manager
    4.3. Obtain batched training dataset
    4.4. Run the training step for each epoch to process all training dataset

---

## GCP:
epochs: 200 / samples: 57k
Epoch 200 Loss 0.6950 Accuracy 0.8074

+ de verdad crees que ella sabe todo?.
+ ella salió de la habitación y se encontró con una escena terrible.
+ no se como agradecerte todo lo que has hecho por mi.
+ deberías echar un vistazo a ese piso.

- no se por donde empezar. todo ha sido muy extraño.
- vamos hombre, ya es hora de que salgas de la cama.
- esta mañana hablé con ella, fue un poco extraño.
- si no vuelvo en cinco minutos, llamad a la policía.