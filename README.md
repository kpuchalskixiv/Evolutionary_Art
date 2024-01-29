# EA project for winter 2022/23 
Jupyter notebook project.ipynb should run after installing all the requirements, feel free to skip cells regarding style transfer.

# Rewriting to CUDA implementation in winter 23/24

Description in Polish:

Program powinien kompilować się po komendzie ‘make’ wykonanej w katalogu ‘cuda’. Później należy uruchomić plik wykonywalny generowany przez make, ‘ES’.

Projekt dotyczy odwzorowywania czarnobiałego obrazu danego na wejściu jako wiele kwadratów nałożonych na siebie, przy pomocy strategii ewolucyjnej. Algorytm dysonuje populacją, tablica floatów [0,1] rozmiaru POPULATION_SIZE X SQUARES_PER_MATE X 5, gdzie 5 wartości per kwadrat odpowiada kordynatom (x,y) na obrazie, rozmiarowi kwadratu, przejrzystości i kolorowi. Strategia działa tylko z mutacją, która jest zaimplementowana za pomocą sigm, jedna wartość sigma dla każdego parametru osobnika, więc rozmiar wektora jest identyczny do wektora populacji co umożliwia proste zrównoleglenie procesu. Najpierw mutowane są sigmy, następnie na ich podstawie kilka razy mutowany jest każdy z wybranych rodziców aby stworzyć populację dzieci. Następnie przeprowadzana jest ewaluacja (rysowanie kwadratów i kalkulacja MSE), po niej pozostaje posortowanie tablicy z wynikami (tablica floatów rozmiaru POP_SIZE, więc niewielka) co wykonywane jest na CPU. Na jej podstawie wyliczam fitness values dzięki którym w sposób probabilistyczny wybieram populację dla kolejnej iteracji.

Przy takich samych hyperparametrach (rozmiar populacji itd.), program na GPU wykonuje ~36x więcej iteracji w tym samym czasie, pomimo wykonywania większej ilości obliczeń niż wersja na CPU. 
