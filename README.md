# StatJAX

## 1.introduce
### 1.1 StatJAX positioning
+ Stat jax is a statistical model algorithm package based on the jax technology stack starting from the matrix calculation level and implemented from the bottom up. I initiated this open source project for the following three purposes:
	+ Real knowledge comes from practice, and you can deepen your understanding of statistical model algorithms by "reinventing the wheel".
	+ Stand on the shoulders of giants and take a small step forward. Based on the JAX technology stack, expand the implementation boundaries of well-known statistical algorithm packages such as statsmodels and scikit learn (such as GPU implementation)
	+ It provides a learning reference for students of probability and statistics.

## 2.design
### 2.1 core concepts
+ **base (module base class)**: base is a module base class implemented using metaprogramming technology. Its main function is to standardize the method behavior of each module, so that different algorithm engineers can jointly develop research algorithms under the same standard, and agile algorithm operation and maintenance and development.
+ **component (algorithm component)**: component is the specific implementation of each algorithm, inherited from the base module base class (different algorithm base class specifications have different method behaviors), its main function is used to implement specific algorithms, and is the main operation, maintenance and development part of the algorithm engineer, and can be independently opened.
+ **algo (application component)**: algo is an open component that combines various algorithm components based on mixins. Its main function is to realize a complete algorithm application flow.

## 3.use
### 3.1 How to get
### 3.2 How to use

## 4.algorithm
### 4.1 Which algorithms and models are covered?

## 5.ChangeLog
<details>
<summary>Click to view ChangeLog</summary>

### Version-0.1.1
  - Algorithm package framework development