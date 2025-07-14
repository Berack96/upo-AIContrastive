# upo-AIContrastive
Progetto per la tesi magistrale.\
Utilizzo di Loss Contrastive per migliorare le predizioni di un Classificatore.

> [!IMPORTANT]
> La versione di Python usata è la [3.10.12](https://www.python.org/downloads/release/python-31012/).\
> Per installare usare il comando: `pip install -r requirements.txt`

I files si trovano dentro la directory [src](src/) e sono tutti dei python notebook.\
Dentro la cartella [keras](keras/) invece si possono trovare degli esempi di keras usati per costruire le reti.

### Cose Fatte
  - [X] Autoencoder: Architettura 3/4 Conv + dense
  - [X] Autoencoder: L’embedding di 128 valori. (messo come parametro iniziale)
  - [X] Autoencoder: Migliorato usando l'architettura VAE
  - [X] Classifier: MLP con BatchNorm ad ogni layer
  - [X] Correction: MLP con BatchNorm ad ogni layer
  - [X] Siamese: Rete che prende in input 2 embedding per passarli alla rete Correction
  - [X] Siamese: Implementazione della loss contrastive semplice
  - [X] Siamese: Implementazione della [Soft-Nearest Neighbors Loss](https://lilianweng.github.io/posts/2021-05-31-contrastive/#soft-nearest-neighbors-loss)
  - [X] Siamese: Implementazione della [Sigmoid Contrastive Loss](https://openreview.net/pdf?id=8QCupLGDT9)

### TODO
Possibili implementazioni/migliorie dei modelli che mi vengono in mente, possono essere combinate o usate singolarmente.\
Li metto qui sotto dato che poi c'è il rischi di dimenticarsele:
- [ ] Usare SVM per il classificatore
- [ ] Usare la loss_contrastive per distanziare le embedding _prima_ di addestrare il classificatore (questo perchè la rete nuova modifica troppo(?) gli embedding e il classificatore non capisce quello che viene messo in input)
