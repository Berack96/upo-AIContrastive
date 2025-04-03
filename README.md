# upo-AIContrastive
Progetto per la tesi magistrale.\
Utilizzo di Loss Contrastive per migliorare le predizioni di un Classificatore.

> [!IMPORTANT]
> La versione di Python usata è la [3.10.12](https://www.python.org/downloads/release/python-31012/).\
> Per installare usare il comando: `pip install -r requirements.txt`

### Cose Fatte
  - [X] Autoencoder: Architettura 3/4 Conv + dense
  - [X] Autoencoder: L’embedding di 256/64 valori. (messo come parametro iniziale)
  - [X] Classifier: Semplice MLP con in fondo un BatchNorm
  - [X] Correction: Semplice MLP denso autoencoder con BatchNorm
  - [X] Siamese: Rete che prende in input 2 embedding per passarli alla rete Correction
  - [X] Siamese: Implementazione della loss contrastive semplice oppure [Soft-Nearest Neighbors Loss](https://lilianweng.github.io/posts/2021-05-31-contrastive/#soft-nearest-neighbors-loss)

### TODO
Possibili implementazioni/migliorie dei modelli che mi vengono in mente, possono essere combinate o usate singolarmente.\
Li metto qui sotto dato che poi c'è il rischi di dimenticarsele:
- [ ] Nel caso usare SVM per il classificatore
- [ ] Modificare la funzione di make_pairs in modo da creare solamente le coppie pensate inizialmente; ovvero prima faccio la classificazione, poi prendo le classi non classificate correttamente e creo una coppia+ andando a prendere una classe classificata correttamente, e una coppia- andando a prendere una classe opposta classificata correttamente
- [ ] Usare la loss_contrastive per distanziare le embedding _prima_ di addestrare il classificatore (questo perchè la rete nuova modifica troppo(?) gli embedding e il classificatore non capisce quello che viene messo in input)
- [ ] Usare un VAE per l'autoencoder in modo da evitare che piccole modifiche dello spazio latente possano influenzare troppo la classificazione. In questo modo magari la rete contrastive può modificare gli embedding senza andare a "confondere" il classificatore
