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
| won_date      | Identificador único do cliente, pode ser o CPF, por exemplo |
| business_segment | Segmento do negócio |
| lead_type| Tipo do lead |
| lead_behaviour_profile| Perfil de comportamento do lead identificado pelo SDR |
|has_company | O lead tem uma empresa com documentação formal?  |
| has_gtin| O lead tem número de item comercial global (código de barras) para seus produtos? |

---

## 3. Feature Engineering


---

## 4. Análise exploratória dos dados

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
