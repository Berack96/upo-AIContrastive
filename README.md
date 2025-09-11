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
  - [X] Siamese: Implementazione della [Soft-Nearest Neighbors Loss](https://lilianweng.github.io/posts/2021-05-31-contrastive/#soft-nearest-neighbors-loss)\
      $L_{SNN} = - \tfrac{1}{|B|} \sum_{x_{i} \in B} log \tfrac{\sum_{(x_{i}, x_{j}) \in D_{i}^{+}} e^f(x_{i},x_{j})/\tau}{\sum_{(x_{i}, x_{j}) \in (D_{i}^{+} \cup D_{i}^{-})} e^f(x_{i},x_{j})/\tau}$
  - [X] Siamese: Implementazione della [Sigmoid Contrastive Loss](https://openreview.net/pdf?id=8QCupLGDT9)\
      $L_{SL} = - \tfrac{1}{|B|}\sum_{i \in B} \sum_{j \in B (j \neq i)} log \tfrac{1}{1 + e^{z_{ij}(-tf(x_{i},x_{j})+b)}}$
  - [X] Classifier: Riaddestramento del classificatore per controllare se la tecnica funziona

### Osservazioni
  - Il metodo di scelta di stop è fatto dopo N epoche e il modello salvato è quello che ha performato meglio nel validation
  - Il dataset `covid_cxr` ha le classi sbilanciate (una è 2/3 volte più frequente)
  - Il dataset `covid_cxr` ha il test molto complicato, tanto che il classificatore non riesce ad essere migliore di un random 50/50
  - Modificato il dataset `covid_cxr` in modo da avere un test più facile spostando degli esempi dal test e dal validation, inoltre le classi sono state bilanciate facendo un sampling.
  - Aggiunto una modalità di creazione degli embedding sintetica.
  - Il dataset `syntetic` è fin troppo facile da predirre e quindi ho modificato la logica in modo da 'avvicinare' le rappresentazioni, rendendole più simili.
  - Dalla visione del TSNE la tecnica di contrastive sembra funzionare, dato che sembra separare bene i vari embedding 
  - Purtroppo però, praticamente sempre dopo l'addestramento contrastive, il classificatore sbaglia prevedendo o una o l'altra classe usando i nuovi embedding.
  - Riaddestrando il classificatore sui nuovi embedding, non riesce comunque ad essere migliore di come era inizialmente
  - Nonostante i continui tentativi, la tecnica sembra essere inefficace.