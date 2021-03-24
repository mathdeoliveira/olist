# Olist - Projeto de um comércio eletrônico

## Índice
- [Introdução](#introdução)
- [1. Sobre o negócio](#1-sobre-o-negócio)
- [2. Sobre os dados](#2-sobre-os-dados)
- [3. Feature Engineering](#3-feature-engineering)
- [4. Análise exploratória dos dados](#4-análise-exploratória-dos-dados)
- [5. Preparação dos dados](#5-preparação-dos-dados)
- [6. Machine Learning - Time Series](#6-machine-learning-time-series)
- [6.1. Performance e resultados](#6-1-performance-resultados)
- [7. Machine Learning - Regressão](#7-machine-learning-regressão)
- [7.1. Performance e resultados](#7-1-performance-resultados)
- [8. Machine Learning - Sistema de Recomendação](#8-machine-learning-sistema-recomendação)
- [8.1. Performance e resultados](#8-1-performance-resultados)
- [9. Teste e monitoramento dos modelos](#9-testes-monitoramento)
- [10. Deploy](#10-deploy)
- [11. Conclusão](#11-Conclusão)

---

## Introdução

Esse projeto tem como intuito o aprendizado e experiência com diversas técnicas de Ciência de Dados. Após a aquisição de conhecimento a partir de diversos cursos, esse projeto vai servir para a aplicação do conhecimento em prática.

O projeto seguirá o método de desenvolvimento CRISP-DS, onde teremos os seguintes processos:
- Entendimento do negócio
- Aquisição dos dados
- Limpeza do dados
- Análise exploratória dos dados
- Preparação dos dados
- Modelagem do machine learning
- Avaliação de performance
- Deploy

**É IMPORTANTE RESSALTAR QUE VOCÊ DEVE TER O AUTENTICADOR DA API DO KAGGLE PARA RODAR O PROCESSO, PARA SABER COMO FAZER ISSO, [CLIQUE AQUI](https://python.plainenglish.io/how-to-use-the-kaggle-api-in-python-4d4c812c39c7)**

---

## 1. Sobre o negócio
**Olist**

Olist é uma startup brasileira que atua no segmento de e-commerce, sobretudo por meio de marketplace. Ela ajuda vendedores a anunciar os seus produtos em grandes sites e-commerce, como por exemplo, Mercado Livre, B2W, Amazon, etc.

Mesmo que temos um conjunto de dados muito bem organizado e com bastante dados temos poucas informações sobre o que fazer, e para contornar esse problema, vamos levantar um cenário hipotético para que assim podermos trabalhar com uma estratégia e objetivo definido.

**Estratégia:**
A empresa necessita de desenvolver sistemas mais inteligentes, entender melhor os seus clientes, ofertar produtos certos para os clientes, prever as vendas futuras e dar uma visão preditiva para os seus vendedores. 

**Objetivo:**
Desenvolver diversos sistema inteligentes baseado em dados para prover a empresa as informações desejadas para aumentar a sua.

**Entregáveis**
- Análise completa dos dados.
- Modelo de séries temporais.
- Modelo de regressão.
- Sistema de recomendação.
- Testes dos modelos.
- Monitoramento dos modelos em tempo real.
- API para cada modelo

---

## 2. Sobre os dados

Os dados utilizados nesse projeto foram disponibilizado pela própria empresa no [Kaggle](https://www.kaggle.com/). 
Sendo o [primeiro dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce) dados de mais de 100k vendas entre 2016 e 2018, onde podemos observar essas vendas e diferentes perspectivas, 
como o valor vendido, status do pedido, frete, localização do cliente, etc. Lembrando que esses dados são reais, porém eles foram anonimizados para proteção dos dados.

E para nós ajudar ainda mais, Olist liberou também o funil de marketing, [nesse link](https://www.kaggle.com/olistbr/marketing-funnel-olist), 
e com ele temos os dados do funil de marketing dos vendedores que preencheram a solicitação para Olist vender os seus produtos na sua loja e com 
eles podemos visualizar nas seguintes perspectivas: categoria do leads, tamanho do catálogo, comportamento, etc.

### 2.1. Tamanho dos nossos dados

Para o primeiro conjunto de dados, o e-commerce, temos 118.315 linhas e 39 variáveis. Já o nosso segundo dataset, temos 12.664 linhas e 23 variáveis.

### 2.2. Estatística descritiva

Dado que temos muitas variáveis, iremos iniciar uma análise descritiva para entender melhor como elas se comportam.

#### 2.2.1. Informações dos dataset

![](https://github.com/mathdeoliveira/olist/blob/dev/notebooks/images/ecommerinfo.PNG?raw=true)

A imagem acima nós da uma informação completa acerca do dataset, vemos que existe alguns valores nulos em algumas variáveis e o tipo que elas se encontram no momento. Já podemos afirmar que vamos precisar lidar com dados nulos, então podemos escolher as linhas nulas ou aplicar algum outro método e também temos que vamos fazer algumas transformações nos tipos das variáveis, como os campo de data por exemplo.

![](https://github.com/mathdeoliveira/olist/blob/dev/notebooks/images/mktinfo.PNG?raw=true)

Continuando no segundo dataset, temos as todas informações sobre as variáveis do marketing funil e novamente vamos lidar com dados faltantes e transformação de alguns dados.

#### 2.2.1. Descrição das variáveis númericas

![](https://github.com/mathdeoliveira/olist/blob/dev/notebooks/images/ecommerdescribe.PNG?raw=true)

Acima temos informações estatísticas sobre as variáveis, existe algumas variáveis que não tem motivos analisar, como o código do CEP(customer_zip_code_prefix) ou campos ID, mas temos que, os clientes divide as suas compras em 3x em média, houve alguma order onde o valor foi de $13.664, lembrando que uma order pode ter múltiplas linhas e temos que a nota que os clientes dão aos produtos tem em média 4 estrelas.

![](https://github.com/mathdeoliveira/olist/blob/dev/notebooks/images/mktdescribe.PNG?raw=true)

Tirando as informações das variáveis identificadora, podemos ver que existe um número altíssimo sobre o que o vendedor declarou para a Olist e os preços dos produtos que querem anunciar é de $134 em média.

![](https://github.com/mathdeoliveira/olist/blob/dev/notebooks/images/ecommerhist.PNG?raw=true)

Os gráficos acima é um histograma para algumas variáveis do dataset e-commerce, vemos que nenhuma segue uma distribuição normal, dependendo do algoritmo escolhido, isso pode ser um dos motivos de a performance ser baixa. Podemos olhar que a variável **price** a grande maioria dos dados estão concentrado entre $0 até $1000.

#### 2.2.2. Descrição das variáveis categóricas

![](https://github.com/mathdeoliveira/olist/blob/dev/notebooks/images/ecommercat.PNG?raw=true)

Nas análises acima, olhamos para as nossas variáveis númericas, a tabela acima olha para algumas variáveis categóricas. Temos algumas informações interessantes, os clientes e vendedores estão concentrados na cidade de São Paulo e no estado de São Paulo, o tipo de pagamento mais utilizado é o cartão de crédito e a categoria mais vendida é a cama, mesa e banho.

![](https://github.com/mathdeoliveira/olist/blob/dev/notebooks/images/mktcat.PNG?raw=true)

Para o dataset sobre funil de marketing, temos que o tipo que mais requisita à Olist a vender o seus produtos é do tipo revendedor, o segmento é de Saúde & Beleza e a origem do tipo de mídia onde foi adquirido foi de pesquisa orgânica. 

### 2.3. Dicionário dos dados
É de suma importância saber o que cada variável representa para o negócio, essa seção vem para mostrar alguma descrição sobre cada variável dos nossos dataset, para não ocupar muito espaço irei retirar as colunas de identificação, siga a tabela a seguir:

#### 2.3.1. Dataset e-commerce 
| Nome variável        | Descrição           |
| ------------- |:-------------:| 
| customer_unique_id      | Identificador único do cliente, pode ser o CPF, por exemplo |
| customer_zip_code_prefix      | CEP      | 
| customer_city | Cidade do cliente     |
| customer_state | Estado do cliente |
| shipping_limit_date | Data limite de envio do vendedor |
| price | Preço do item |
| freight_value | Preço do frente, se uma order tiver mais de um item, o preço é dividido entre elas |
| payment_sequential | Cliente pode pagar com mais de um método de pagamento, se ele fizer uma sequência é criada |
| payment_type | Método de pagamento escolhido pelo cliente |
| payment_installments | Número de parcelas |
| payment_value | Valor da transação |
| review_score | Nota de 1 até 5 sobre a sua satisfação|
| review_comment_title | Título do review |
| review_comment_message | Mensagem deixada pelo cliente |
| review_creation_date   | Data de envio da pesquisa |
| review_answer_timestamp | Data da resposta da pesquisa pelo cliente |
| order_status | Status do pedido |
| order_purchase_timestamp | Data da compra |
| order_approved_at | Data da aprovação do pagamento |
| order_delivered_carrier_date | Data de quanto foi entregue a transportadora |
| order_delivered_customer_date | Data de envio para o cliente |
| order_estimated_delivery_date | Data da estimativa da entrega |
| product_category_name | Categoria raiz do produto |
| product_name_lenght   | Número de caracteres do nome do produto |
| product_description_lenght |  Número de caracteres da descrição do produto|
| product_photos_qty | Número de fotos publicadas do produto |
| product_weight_g | Peso do produto em gramas |
| product_length_cm | Comprimento do produto em centímetros|
| product_height_cm | Altura do produto em centímetros |
| product_width_cm | Largura do produto em centímetros |
| seller_zip_code_prefix | CEP do vendedor |
| seller_city | Cidade do vendedor  |
| seller_state | Estado do vendedor  |

#### 2.3.2. Dataset marketing funil
| Nome variável        | Descrição           |
| ------------- |:-------------:| 
| won_date      | Data que o negócio foi fechado |
| business_segment | Segmento do negócio |
| lead_type| Tipo do lead |
| lead_behaviour_profile| Perfil de comportamento do lead identificado pelo SDR |
|has_company | O lead tem uma empresa com documentação formal?  |
| has_gtin| O lead tem número de item comercial global (código de barras) para seus produtos? |
| first_contact_date | Data da primeira solitação |
|origin | Tipo de mídia onde o lead foi adquirido |

---

## 3. Feature Engineering

Afim de potencializar a nossa análise exploratória dos dados foi criado o mapa mental abaixo, com ela podemos nos guiar para criar hipóteses baseado no 
negócio em si, retirando o maior número de insights possíveis com os dados que possuimos.

![](https://github.com/mathdeoliveira/olist/blob/dev/notebooks/images/mind_map.jpg?raw=true)

### Hipóteses
#### Hipóteses clientes

- Clientes que tiveram os seus pedidos enviados após a data da estimativa de entrega não voltam a comprar
- Clientes de SP são os que tem o maior volume de compra
- Clientes que compram com cartão de crédito compra mais que do que aqueles que compram com boleto
- Clientes que tiveram as suas compras canceladas não voltam a comprar
- Clientes que pagam alto valor de frete compram diversos produtos

#### Hipóteses produtos
- Produtos com peso abaixo do primeiro quartil são os mais vendidos em termos de quantidade
- Categoria do produto mais vendido é aquela relacionada a eletrônicos, pois o valor agregado é alto
- Os produtos que tem mais fotos publicadas são os mais vendidos em termos de valor
- Produtos com nome longos são menos vendidos
- Produtos com descrição longas são os mais vendidos
- Produtos com grandes dimensões(altura, peso, largura...) são menos vendidos, porém tem alto retorno pro negócio

#### Hipóteses vendedores
- Vendedores de SP são os que mais atrasam os envios
- Os melhores vendedores, de acordo com a nota do review, são os que mais vendem produtos

#### Hipóteses pagamentos
- Todas as compras feitas foram parceladas em 2x
- Quanto mais alta são as notas dos reviews menor as parcelas
- Cartão de crédito é o método de pagamento que mais atrasa o envio para transportadora

#### Hipóteses temporal
- A maioria das compras são feitas antes do dia 15
- Há um aumento no valor de vendas aprovadas no mês de dezembro

#### Hipóteses funil de marketing
- Os leads que não fecharam o negócio são os que foram adquiridos por pesquisa orgânica
- Os leads que tem uma empresa formal são a maioria que fecham o negócio

#### Substituir valores faltantes

As colunas **categóricas**:
 - Para as colunas relacionadas ao review, como o título e a mensagem, vamos deletar essas duas colunas pois nesse projeto não iremos lidar com NLP. 
 - Para o nome da categoria do produto vamos substituir os valores faltantes por 'missing'.
 
As colunas do tipo **data**:
 - Essas linhas serão deletadas, já que representam uma porcentagem bem pouca sobre todo o dataset.

Para as colunas **númericas**:
- adicionar uma coluna indicando que existia valores faltantes
- substituir os valores faltantes pela moda na coluna original

---

## 4. Análise exploratória dos dados

### Análise univariada

Essa análise vamos descrever algumas variáveis examinando a distribuição dos casos de apenas uma variável e cada vez.

O gráfico abaixo nos mostra a distribuição da variável payment_value, que foi o valor da transação da venda, vemos que está bem concentrado em valores aaixo de $1.000 e vimos que na seção [sobre os dados](#2-sobre-os-dados) na parte da estatística descritiva existe valores outliers.


Os status das orders dos nossos dados estão bem desbalanceados, como vemos abaixo, a grande maioria das orders já foram enviadas e um valor muito pequeno foram canceladas, com isso podemos ter algum problema a frente.


O tipo de pagamento mais utilizado é cartão de crédito, como vemos abaixo, mas vemos que existe compras feitas por voucher, boleto e cartão de débito, também temos que pensar nessa proporção pequena de dados para cartão de débito.


Os reviews das vendas são bem positivos, a maioria está entre as notas 4 e 5, e interessante ver que a nota 5 é mais que o dobro maior que a nota 4.


A categoria de produtos tem a maior quantidade de orders é a Cama & Mesa & Banho, não muito atrás, as outras seguem bem próximas das maiores do que e.a


### Análise bivariada

A análise bivariada nos permite a duas variáveis de forma simultânea, por isso vamos usar-la nesse projeto para responder as nossas hipóteses levantadas anteriormente.

**H1. Clientes que tiveram os seus pedidos enviados após a data da estimativa de entrega não voltam a comprar - FALSA**


No gráfico temos que existe sim clientes que receberam os seus produtos após a data de estimativa, então houve algum tipo de atraso. Como a nossa hipótese levantada diz que os clientes tiveram os seus pedidos enviados após a data da estimativa de entrega não voltam a comprar, vamos capturar um exemplo da base onde o cliente comprou e teve atraso e após esse atraso voltou a comprar.

Vemos que na tabela existe um cliente que comprou com atraso e voltar a comprar logo após a compra, assim a hipótese é FALSA.

**H2. Clientes de SP são os que tem o maior volume de compra - VERDADEIRA**

Fica evidente que a hipótese é verdadeira, tanto é que, a cidade RJ tem 63% menos em volume de venda do que SP. Em outras palavras, SP vende mais que o dobro da segunda colocada.

**H3. Clientes que compram com cartão de crédito compra mais que do que aqueles que compram com boleto - VERDADEIRA**

Novamente, é uma hipótese verdadeira, vemos que o método de pagamento cartão de crédito é o mais utilizado do que os outros, do total de venda de 20.418.288,15, o cartão de crédito foi responsável por 15.670.920,67 das vendas da empresa representando 76% do total.

**H4. Clientes que tiveram as suas compras canceladas não voltam a comprar - VERDADEIRA**


Essa hipótese é verdadeira, porém devemos levar em consideração que existe uma quantidade bem pouca sobre os clientes que cancelaram suas compras, das 115.711 orders, somente 7 clientes tem status cancelado. Por isso devemos capturar mais dados para realmente provar essa hipótese.

**H5. Clientes que pagam alto valor de frete compram diversos produtos - VERDADEIRA**


Hipótese verdadeira, mesmo que tenha algumas quedas na quantidade de produtos vs total pago pelo frete, quanto maior a quantidade de produtos distintos comprados maior é o valor do frete.


**H6. Produtos com peso abaixo do primeiro quartil são os mais vendidos em termos de quantidade - FALSA**

Sabemos que o primeiro quartil para a variável é ate 300 gramas, para tal, tivemos 28.071 vendas dos produtos abaixo de 300 gramas e para acima de 300 gramas tivemos 83.316, por isso a hipótese é falsa, produtos mais pesados que 300 gramas tem a maior quantidade vendida.


**H7. Categoria do produto mais vendido é aquela relacionada a eletrônicos, pois o valor agregado é alto - VERDADEIRA***

Para os dez primeiros, podemos parcialmente retirar algumas informações já que não temos um campo falando exatamente qual universo o produto se encontra. O produto com o maior valor vendido é relacionado a telefonia fixa, que em maior parte é um eletrônico, também vemos a categoria informática e acessórios, que também é eletrônicos, podemos confirmar essa hipótese, mas com atenção mostrando que existe outras categoria que também vende bem.

**H8. Os produtos que tem mais fotos publicadas são os mais vendidos em termos de valor - FALSA**

Hipótese falsa, não há evidências suficientes que quanto maior a quantidade de fotos publicada do produto, maior é o valor da venda, o gráfico ilustra uma queda no total vendido enquanto a quantidade de fotos publicadas aumenta, pode levar em consideração que o produto que tem 20 fotos publicadas teve total de vendas abaixo do que o produto com 19 fotos publicada.


**H9. Produtos com nome longos são menos vendidos - FALSA**

Hipótese falsa, existe exemplos de produtos com nomes longos onde vendeu uma quantidade bem alta.

**H10. Produtos com descrição longas são os mais vendidos - FALSA**

Hipótese falsa, quanto maior o tamanho da descrição do produto menor é a quantidade vendida desse produto.

**H11. Produtos com grandes dimensões(altura, peso, largura...) são menos vendidos, porém tem alto retorno pro negócio - VERDADEIRA**

Definindo que, produtos com grandes dimensões:
- Peso acima de 10000 gramas (10kg)
- Comprimento acima de 60cm
- Altura acima de 60cm
- Largura acima de 60cm

Dado isso temos que, a hipótese é verdadeira, vemos no gráfico que os produtos grandes tem uma baixa quantidades vendidas mas tem um valor total maior que o restante dos produtos.

**H12. Vendedores de SP são os que mais atrasam os envios - VERDADEIRA**

Hipótese verdadeira, os vendedores do estado de SP são os vendedores que mais atrasam o pedido do cliente.

**H13. Os melhores vendedores, de acordo com a nota do review, são os que mais vendem produtos - FALSA**

Hipótese falsa, vemos que existe um comportamento onde quanto maior a nota do vendedor, menor é a quantidade de produtos vendidos por ele, como exemplo os vendedores com nota entre 4,5 e 5, vendem menos que os vendedores com nota 4. Podemos explica isso também pela quantidade de vendedores que temos com nota maior que 4,5.

**H14. Todas as compras feitas foram parceladas em 2x - FALSA**

Hipótese falsa, as compras são feitas parceladas em diversas parcelas, e fica evidente que a maioria das parcelas não são parceladas, somente 1x,

**H15. Quanto mais alta são as notas dos reviews menor as parcelas - FALSA**

Hipótese falsa, não há evidências o suficiente para confirmar que quanto mais alta são as notas dos reviews menor as parcelas, vemos que não importa a nota do review, vamos ter compras com diversas quantidade de parcelas.


**H16. Cartão de crédito é o método de pagamento que mais atrasa o envio para transportadora - VERDADEIRA**

Hipótese verdadeira, vemos que o cartão de crédito é o método de pagamento que mais atrasa o envio para transportadora, isso é resultado de que a maioria das compras feitas na Olist são feitas com cartão de crédito, crescendo a quantidade de atrasos.

**H17. A maioria das compras são feitas antes do dia 15 - VERDADEIRA**

Hipótese verdadeira, por bem pouco, as vendas são feitas antes do dia 15, porém bem apertado para as vendas após o dia 15, atenção nessa hipótese pois pode ser indício de poucos dados para tomar essa decisão.

**H18. Há um aumento no valor de vendas aprovadas no mês de dezembro - FALSA**

Hipótese falsa, dentro de todos os meses dos anos que faz parte do nosso dataset, o mês de Dezembro é onde temos uma queda de vendas, comparado com o mês anterior. Podemos ver também que dezembro é o segundo mês que tem o menor valor total de vendas, somente atrás de setembro.

**H19. Os leads que não fecharam o negócio são os que foram adquiridos por pesquisa orgânica - FALSA**

Hipótese falsa, mesmo que a origem dos leads que ainda não fecharam o negócio é pesquisa orgânica, ainda tem outras negócios não fechados para outras origens.

**H20. Os leads que tem uma empresa formal são a maioria que fecham o negócio - VERDADEIRA**

Hipótese verdadeira, a maioria dos vendedores que pedem requisição para virar parceiro da empresa, a maioria que fecham o negócio são vendedores que possuem empresa formal.


---

## 5. Preparação dos dados

---

## 6. Machine Learning - Time Series

---

### 6.1. Performance e resultados

---

## 7. Machine Learning - Regressão

---

### 7.1. Performance e resultados

---

## 8. Machine Learning - Sistema de Recomendação

---

### 8.1. Performance e resultados

---

## 9. Teste e monitoramento dos modelos

---

## 10. Deploy

---

## 11. Conclusão
