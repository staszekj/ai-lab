# Pochodna funkcji straty względem wagi (eng: loss derivative with respect to a weight)

Pochodną (eng: derivative) funkcji $f$ w punkcie $x$ definiujemy jako granicę
ilorazu różnicowego (eng: difference quotient):

$$
f'(x) = \lim_{h \to 0}\frac{f(x+h)-f(x)}{(x+h)-x}.
$$

## Gradient (eng: gradient)

Najpierw rozważmy funkcję jednej zmiennej:

$$
f(w) = (5w-10)^2.
$$

Wprowadzamy funkcję pomocniczą $g(w)=5w-10$, więc $f(w)=g(w)^2$. Pochodna
(eng: derivative) funkcji $f$ względem $w$ wynika z reguły łańcuchowej:

$$
\frac{df}{dw}
= \frac{df}{dg}\frac{dg}{dw}
= 2g(w) \cdot 5
= 10(5w-10).
$$

Dla jednej zmiennej gradient jest właśnie tą pochodną. Gdy $w=3$, pochodna
ma wartość dodatnią, więc aby zmniejszyć $f$, należy zmniejszyć $w$. Gdy
$w=1$, pochodna jest ujemna, więc należy zwiększyć $w$. Gradient wskazuje
kierunek najszybszego wzrostu funkcji, dlatego minimalizacja wykonuje krok
w kierunku przeciwnym do gradientu.

## Wymiary mnożenia macierzy (eng: matrix multiplication shapes)

Poniższe przykłady pokazują wyłącznie wymiary. Wektor po lewej stronie
traktujemy jako wektor wierszowy, a wektor po prawej jako wektor kolumnowy.
Po prawej stronie każdego równania znajduje się odpowiadający mu zapis
kształtu tensora w PyTorch, na przykład `(3,)` dla wektora o trzech elementach.

$$
\underbrace{\mathbf{u}}_{1 \times n}
\mathbin{@}
\underbrace{\mathbf{v}}_{n \times 1}
=
\underbrace{s}_{1 \times 1}
\qquad \texttt{PyTorch: (3,) @ (3,) -> ()}.
$$

$$
\underbrace{\mathbf{u}}_{1 \times n}
\mathbin{@}
\underbrace{A}_{n \times m}
=
\underbrace{\mathbf{w}}_{1 \times m}
\qquad \texttt{PyTorch: (3,) @ (3, 5) -> (5,)}.
$$

$$
\underbrace{A}_{m \times n}
\mathbin{@}
\underbrace{B}_{n \times p}
=
\underbrace{C}_{m \times p}
\qquad \texttt{PyTorch: (5, 3) @ (3, 4) -> (5, 4)}.
$$

Mnożenie jest możliwe, gdy wewnętrzne wymiary są równe: $n$ w pierwszym
argumencie musi odpowiadać $n$ w drugim.

## Schemat sieci (eng: network diagram)

Macierze wag wyznaczają granice kolejnych warstw. Po lewej stronie wchodzą
sygnały poprzedniej warstwy, a po prawej wychodzi inna liczba sygnałów.
Każda warstwa ukryta stosuje funkcję aktywacji (eng: activation function)
elementowo, a softmax (eng: softmax) przekształca cały wektor logitów
(eng: logits) na wyjściu sieci.

```text
 wejscie                                                                                          wyjscie

                              +----------+ ----------------> z_1^(l-1) -> [--ReLU--] -> h_1^(l-1)
 x_1 -----------------------> |          | ----------------> z_2^(l-1) -> [--ReLU--] -> h_2^(l-1)
 x_2 -----------------------> | W^(l-1)  | ----------------> z_3^(l-1) -> [--ReLU--] -> h_3^(l-1)
 x_3 -----------------------> |          | ----------------> z_4^(l-1) -> [--ReLU--] -> h_4^(l-1)
                              +----------+ ----------------> z_5^(l-1) -> [--ReLU--] -> h_5^(l-1)

 h_1^(l-1) -----------------> +----------+ ----------------> z_1^(l) -> [--GELU--] -> h_1^(l)
 h_2^(l-1) -----------------> |          | ----------------> z_2^(l) -> [--GELU--] -> h_2^(l)
 h_3^(l-1) -----------------> | W^(l)    | ----------------> z_3^(l) -> [--GELU--] -> h_3^(l)
 h_4^(l-1) -----------------> |          | ----------------> z_4^(l) -> [--GELU--] -> h_4^(l)
 h_5^(l-1) -----------------> +----------+

 h_1^(l) -------------------> +----------+ ----------------> z_1^logits -+                    +-> p_1
 h_2^(l) -------------------> |          | ----------------> z_2^logits -+---- [--softmax--] -+-> p_2
 h_3^(l) -------------------> | W^out    | ----------------> z_3^logits -+                    +-> p_3
 h_4^(l) -------------------> +----------+

 p_1, p_2, p_3 -------------------------------------------------> L
```

W przykładzie warstwy mają odpowiednio $3$, $5$, $4$ i $3$ składowe. Funkcja
aktywacji $\varphi$ działa elementowo na wartościach $z^{(r)}$, a softmax działa
wspólnie tylko na wektorze logitów, zwracając prawdopodobieństwa $p_j$.

W rzeczywistej sieci softmax działa na całym wektorze logitów (eng: logits),
a nie niezależnie na każdej linii. Schemat pokazuje najważniejsze zależności:

Wektory $h^{(l-1)}$ i $h^{(l)}$ zawierają odpowiednio wyjścia neuronów
w warstwach ukrytych (eng: hidden layers) $l-1$ i $l$.

$$
x_i \xrightarrow{W^{(l-1)}} z_k^{(l-1)}
\xrightarrow{\varphi} h_k^{(l-1)}
\xrightarrow{W^{(l)}} z_a^{(l)}
\xrightarrow{\varphi} h_a^{(l)}
\xrightarrow{W^{\mathrm{out}}} z_q^{\mathrm{logits}}
\xrightarrow{\mathrm{softmax}} \mathbf{p}
\xrightarrow{\mathrm{cross\mathchar`-entropy}} L.
$$

Rozważmy funkcję straty, czyli entropię krzyżową (eng: cross-entropy):

$$
L = -\sum_{j=1}^{K} y_j \ln(p_j),
$$

gdzie $y_j$ jest wartością docelową, a $p_j$ przewidywanym prawdopodobieństwem klasy $j$.

Pochodna straty względem prawdopodobieństwa klasy $j$ wynosi:

$$
\frac{\partial L}{\partial p_j} = -\frac{y_j}{p_j}.
$$

## Funkcje aktywacji (eng: activation functions)

W pierwszej warstwie ukrytej schemat stosuje ReLU (eng: rectified linear unit):

$$
\operatorname{ReLU}(x) =
\begin{cases}
0 & \text{dla } x \leq 0, \\
x & \text{dla } x > 0,
\end{cases}
\qquad
\operatorname{ReLU}'(x) =
\begin{cases}
0 & \text{dla } x < 0, \\
1 & \text{dla } x > 0.
\end{cases}
$$

W punkcie $x=0$ pochodna ReLU nie istnieje; w praktyce biblioteki zwykle
przyjmują tam wartość $0$.

W drugiej warstwie ukrytej schemat stosuje GELU (eng: Gaussian error linear
unit):

$$
\operatorname{GELU}(x) = x\Phi(x),
\qquad
\operatorname{GELU}'(x) = \Phi(x) + x\phi(x),
$$

gdzie $\Phi(x)$ jest dystrybuantą standardowego rozkładu normalnego
(eng: standard normal cumulative distribution function), a $\phi(x)$ jego
funkcją gęstości (eng: probability density function):

$$
\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}.
$$

## Softmax (eng: softmax)

Prawdopodobieństwa (eng: probabilities) otrzymujemy z logitów (eng: logits):

$$
p_j = \frac{e^{z_j^{\mathrm{logits}}}}
{\sum_{q=1}^{K}e^{z_q^{\mathrm{logits}}}}.
$$

Zatem $\sum_j p_j=1$. Ponieważ softmax jest funkcją wielu wejść, pochodną
$p_j$ względem logitu $z_q^{\mathrm{logits}}$ rozbijamy na dwa przypadki.

Jeżeli różniczkujemy względem wartości wejściowej odpowiadającej temu samemu indeksowi
($q=j$), otrzymujemy:

$$
\frac{\partial p_j}{\partial z_j^{\mathrm{logits}}}
= p_j(1-p_j).
$$

Jeżeli różniczkujemy względem innej wartości wejściowej ($q\neq j$), otrzymujemy:

$$
\frac{\partial p_j}{\partial z_q^{\mathrm{logits}}}
= -p_jp_q.
$$

Indeks $j$ oznacza prawdopodobieństwo, natomiast $q$ logit, względem którego
różniczkujemy.

Jeżeli wartości docelowe tworzą rozkład prawdopodobieństwa, czyli
$\sum_j y_j=1$ (w szczególności są zakodowane one-hot), to po połączeniu
cross-entropy z softmaxem otrzymujemy końcową pochodną straty względem
wejścia softmaxa:

$$
\begin{aligned}
\frac{\partial L}{\partial z_q^{\mathrm{logits}}}
&= \left(-\frac{y_q}{p_q}\right)p_q(1-p_q)
+ \sum_{j\neq q}\left(-\frac{y_j}{p_j}\right)(-p_jp_q) \\
&= p_q - y_q.
\end{aligned}
$$

## Reguła łańcuchowa (eng: chain rule)

Przyjmijmy:

<div align="left">

$$
z_a^{(r)}=\sum_b W_{ab}^{(r)}h_b^{(r-1)},
\qquad
h_a^{(r)}=\varphi(z_a^{(r)}).
$$

</div>

Funkcja $\varphi$ jest aktywacją elementową, więc $h_a^{(r)}$ zależy tylko od
$z_a^{(r)}$.

Najkrótsze postacie pochodnych względem wag warstwy wyjściowej i warstwy $l$
są następujące:

<div align="left">

$$
\begin{aligned}
\frac{\partial L}{\partial W_{q i}^{\mathrm{out}}}
&=
\sum_j
\frac{\partial L}{\partial p_j}
\frac{\partial p_j}{\partial z_q^{\mathrm{logits}}}
\frac{\partial z_q^{\mathrm{logits}}}{\partial W_{q i}^{\mathrm{out}}} \\
&=
(p_q-y_q)h_i^{(l)}.
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial W_{a k}^{(l)}}
&=
\sum_q
\sum_j
\frac{\partial L}{\partial p_j}
\frac{\partial p_j}{\partial z_q^{\mathrm{logits}}}
\frac{\partial z_q^{\mathrm{logits}}}{\partial h_a^{(l)}}
\frac{\partial h_a^{(l)}}{\partial z_a^{(l)}}
\frac{\partial z_a^{(l)}}{\partial W_{a k}^{(l)}}
\\
&=
\sum_q\sum_j
\frac{\partial L}{\partial p_j}
\frac{\partial p_j}{\partial z_q^{\mathrm{logits}}}
W_{q a}^{\mathrm{out}}
\varphi'\bigl(z_a^{(l)}\bigr)
h_k^{(l-1)}.
\end{aligned}
$$

</div>

<div align="left">

$$
\begin{aligned}
\frac{\partial L}{\partial W_{km}^{(l-1)}}
&=\sum_j\sum_q\sum_i
\frac{\partial L}{\partial p_j}
\frac{\partial p_j}{\partial z_q^{\mathrm{logits}}}
\frac{\partial z_q^{\mathrm{logits}}}{\partial h_i^{(l)}}
\frac{\partial h_i^{(l)}}{\partial z_i^{(l)}}
\frac{\partial z_i^{(l)}}{\partial h_k^{(l-1)}}
\frac{\partial h_k^{(l-1)}}{\partial z_k^{(l-1)}}
\frac{\partial z_k^{(l-1)}}{\partial W_{km}^{(l-1)}} \\
&=\sum_j\sum_q\sum_i
\frac{\partial L}{\partial p_j}
\frac{\partial p_j}{\partial z_q^{\mathrm{logits}}}
W_{q i}^{\mathrm{out}}
\varphi'\bigl(z_i^{(l)}\bigr)
W_{i k}^{(l)}
\varphi'\bigl(z_k^{(l-1)}\bigr)
h_m^{(l-2)}.
\end{aligned}
$$

</div>

Jeżeli $x_i=h_i^{(l-2)}$ jest $i$-tym wejściem (eng: input) do sieci,
analogiczny łańcuch dla pochodnej względem tego wejścia ma postać:

<div align="left">

$$
\begin{aligned}
\frac{\partial L}{\partial x_i}
&=\sum_j\sum_q\sum_a\sum_k
\frac{\partial L}{\partial p_j}
\frac{\partial p_j}{\partial z_q^{\mathrm{logits}}}
\frac{\partial z_q^{\mathrm{logits}}}{\partial h_a^{(l)}}
\frac{\partial h_a^{(l)}}{\partial z_a^{(l)}}
\frac{\partial z_a^{(l)}}{\partial h_k^{(l-1)}}
\frac{\partial h_k^{(l-1)}}{\partial z_k^{(l-1)}}
\frac{\partial z_k^{(l-1)}}{\partial x_i} \\
&=\sum_j\sum_q\sum_a\sum_k
\frac{\partial L}{\partial p_j}
\frac{\partial p_j}{\partial z_q^{\mathrm{logits}}}
W_{q a}^{\mathrm{out}}
\varphi'\bigl(z_a^{(l)}\bigr)
W_{a k}^{(l)}
\varphi'\bigl(z_k^{(l-1)}\bigr)
W_{k i}^{(l-1)}.
\end{aligned}
$$

</div>

## Aktualizacja parametrów (eng: parameter update)

Powyższe pochodne można wykorzystać w metodzie stochastycznego spadku
gradientowego (eng: stochastic gradient descent, SGD). W kroku SGD
aktualizujemy wagę według wzoru:

$$
W_{q i}^{\mathrm{out}} \leftarrow W_{q i}^{\mathrm{out}}
- \eta \frac{\partial L}{\partial W_{q i}^{\mathrm{out}}}.
$$

$$
W_{a k}^{(l)} \leftarrow W_{a k}^{(l)}
- \eta \frac{\partial L}{\partial W_{a k}^{(l)}}.
$$

$$
W_{km}^{(l-1)} \leftarrow W_{km}^{(l-1)}
- \eta \frac{\partial L}{\partial W_{km}^{(l-1)}}.
$$

Analogicznie można aktualizować wejście, na przykład $x_n$:

$$
x_n \leftarrow x_n - \eta \frac{\partial L}{\partial x_n}.
$$

Współczynnik $\eta > 0$ jest krokiem uczenia (eng: learning rate).

## AdamW (eng: adaptive moment estimation with decoupled weight decay)

AdamW korzysta z ruchomej średniej (eng: moving average) gradientu
(eng: gradient) oraz jego kwadratu. Dla dowolnego optymalizowanego parametru
$\omega$ i gradientu
$g_t = \partial L / \partial \omega_t$ przyjmujemy wartości początkowe
$m_0=0$ i $v_0=0$, a następnie obliczamy:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t}.
\end{aligned}
$$

Następnie wykonujemy aktualizację z rozdzielonym zanikiem wag
(eng: decoupled weight decay):

$$
\omega_{t+1} \leftarrow
\omega_t - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}
- \eta_t \lambda \omega_t.
$$

Współczynniki $\beta_1$ i $\beta_2$ sterują pamięcią średnich, $\varepsilon$
zapewnia stabilność numeryczną, a $\lambda$ określa siłę weight decay.

## Dynamiczny krok uczenia (eng: dynamic learning rate)

Krok uczenia nie musi być stały: jego wartość w chwili $t$ zapisujemy jako
$\eta_t$. Harmonogram (eng: schedule) może zaczynać od krótkiego rozgrzewania
(eng: warm-up), następnie maleć w czasie treningu, na przykład liniowo lub
zgodnie z harmonogramem cosinusowym (eng: cosine schedule). Takie dynamiczne
dobieranie kroku uczenia pozwala wykonywać większe kroki na początku, a bliżej
minimum funkcji straty wykonywać mniejsze, stabilniejsze aktualizacje.

Znaczenie kolejnych czynników:

1. $\partial L/\partial p_j$ opisuje wpływ prawdopodobieństwa $p_j$ na stratę.
2. $\partial p_j/\partial z_q^{\mathrm{logits}}$ opisuje wpływ logitu $q$ na prawdopodobieństwo $p_j$.
3. $\partial z_q^{\mathrm{logits}}/\partial h_i^{(l)}$ przenosi wpływ do neuronu $i$.
4. $\partial h_i^{(l)}/\partial z_i^{(l)}$ uwzględnia aktywację warstwy $l$.
5. $\partial z_i^{(l)}/\partial h_k^{(l-1)}$ przenosi wpływ do neuronu $k$.
6. $\partial h_k^{(l-1)}/\partial z_k^{(l-1)}$ uwzględnia aktywację najniższej warstwy ukrytej.
7. $\partial z_k^{(l-1)}/\partial W_{km}^{(l-1)}$ uwzględnia wejście $h_m^{(l-2)}$ do różniczkowanej wagi.
