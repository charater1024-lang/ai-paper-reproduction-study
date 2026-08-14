# AI 7개 분야 × 유명 논문 10편 실습

컴퓨터 비전, NLP·LLM, 생성모델, 강화학습·에이전트, 그래프·추천, 자기지도·멀티모달,
지식 증류·모델 경량화에서 10편씩 선정한 **총 70편**의 하드웨어 인식 미니 재현 트랙입니다.
논문마다 실습본과 정답·해설본이 한 쌍이므로 총 140개 노트북이 제공됩니다.

> 이 자료는 원 논문의 대규모 benchmark 전체 재현이 아닙니다. 핵심 수식·구조·알고리즘을
> 로컬 합성 데이터에서 검증하는 교육용 축소 재현입니다.

## 시작

```powershell
.\Start_Field_Paper_Labs.cmd

# 분야와 논문 번호를 직접 지정: NLP·LLM의 03번
.\Start_Field_Paper_Labs.cmd nlp_llm 03
```

[상세 학습법과 검증 명령](../../docs/FIELD_PAPER_LABS_GUIDE.md) · [한국어 논문 독해 노트](../../docs/paper_reading_notes/README.md) · [포트폴리오 구현 표준](../../docs/PORTFOLIO_NOTEBOOK_STANDARD.md) · [로컬 데이터 설명](../../data/field_curriculum/README.md)

## 컴퓨터 비전

| 번호 | 논문 (연도) | 핵심 미니 재현 | 데이터 | 실습 | 정답 |
|---:|---|---|---|---|---|
| 00 | [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) (1998) | 공유 커널과 평균 subsampling으로 C1-S2-C3-S4 경로를 재현한다. | `vision_shapes.npz` | [열기](vision/exercises/00_lenet5.ipynb) | [열기](vision/solutions/00_lenet5.ipynb) |
| 01 | [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) (2012) | ReLU, channel-local LRN, overlapping pooling, 5-conv 경로를 재현한다. | `vision_shapes.npz` | [열기](vision/exercises/01_alexnet.ipynb) | [열기](vision/solutions/01_alexnet.ipynb) |
| 02 | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (2014) | 연속 3×3 convolution의 receptive field와 2/2/3 block을 재현한다. | `vision_shapes.npz` | [열기](vision/exercises/02_vgg.ipynb) | [열기](vision/solutions/02_vgg.ipynb) |
| 03 | [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) (2014) | multi-scale 병렬 분기와 1×1 dimension reduction을 재현한다. | `vision_shapes.npz` | [열기](vision/exercises/03_googlenet_inception.ipynb) | [열기](vision/solutions/03_googlenet_inception.ipynb) |
| 04 | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (2015) | contracting/expanding path와 skip concatenation으로 mask를 분할한다. | `vision_shapes.npz` | [열기](vision/exercises/04_unet.ipynb) | [열기](vision/solutions/04_unet.ipynb) |
| 05 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (2015) | residual mapping과 identity/projection shortcut을 직접 구현한다. | `vision_shapes.npz` | [열기](vision/exercises/05_resnet.ipynb) | [열기](vision/solutions/05_resnet.ipynb) |
| 06 | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) (2015) | RPN anchor, IoU assignment, objectness+box multi-task loss를 재현한다. | `vision_shapes.npz` | [열기](vision/exercises/06_faster_rcnn.ipynb) | [열기](vision/solutions/06_faster_rcnn.ipynb) |
| 07 | [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015) | grid target encoding과 coordinate/object/class 결합 손실을 재현한다. | `vision_shapes.npz` | [열기](vision/exercises/07_yolov1.ipynb) | [열기](vision/solutions/07_yolov1.ipynb) |
| 08 | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (2020) | patch embedding, class token, position embedding, pre-LN encoder를 재현한다. | `vision_shapes.npz` | [열기](vision/exercises/08_vision_transformer.ipynb) | [열기](vision/solutions/08_vision_transformer.ipynb) |
| 09 | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) (2020) | object queries, bipartite matching, no-object class, set loss를 재현한다. | `vision_shapes.npz` | [열기](vision/exercises/09_detr.ipynb) | [열기](vision/solutions/09_detr.ipynb) |

## 자연어 처리와 LLM

| 번호 | 논문 (연도) | 핵심 미니 재현 | 데이터 | 실습 | 정답 |
|---:|---|---|---|---|---|
| 00 | [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (2013) | skip-gram context pair, negative sampling, unigram^(3/4) noise를 재현한다. | `nlp_corpus.json` | [열기](nlp_llm/exercises/00_word2vec_sgns.ipynb) | [열기](nlp_llm/solutions/00_word2vec_sgns.ipynb) |
| 01 | [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (2014) | encoder fixed vector, teacher forcing decoder, reversed source를 재현한다. | `nlp_corpus.json` | [열기](nlp_llm/exercises/01_seq2seq.ipynb) | [열기](nlp_llm/solutions/01_seq2seq.ipynb) |
| 02 | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (2014) | additive alignment energy, softmax weights, context vector를 재현한다. | `nlp_corpus.json` | [열기](nlp_llm/exercises/02_bahdanau_attention.ipynb) | [열기](nlp_llm/solutions/02_bahdanau_attention.ipynb) |
| 03 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) | scaled dot-product, multi-head 분할, sinusoidal position을 직접 재현한다. | `nlp_corpus.json` | [열기](nlp_llm/exercises/03_attention_is_all_you_need.ipynb) | [열기](nlp_llm/solutions/03_attention_is_all_you_need.ipynb) |
| 04 | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018) | causal LM pretraining과 supervised+auxiliary LM fine-tuning을 재현한다. | `nlp_corpus.json` | [열기](nlp_llm/exercises/04_gpt1.ipynb) | [열기](nlp_llm/solutions/04_gpt1.ipynb) |
| 05 | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (2018) | 80/10/10 MLM corruption과 bidirectional encoder를 재현한다. | `nlp_corpus.json` | [열기](nlp_llm/exercises/05_bert.ipynb) | [열기](nlp_llm/solutions/05_bert.ipynb) |
| 06 | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) (2019) | text-to-text 형식, span sentinel target, relative position bias를 재현한다. | `nlp_corpus.json` | [열기](nlp_llm/exercises/06_t5.ipynb) | [열기](nlp_llm/solutions/06_t5.ipynb) |
| 07 | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (2020) | zero/one/few-shot prompt와 parameter update 없는 in-context protocol을 재현한다. | `nlp_corpus.json` | [열기](nlp_llm/exercises/07_gpt3_few_shot.ipynb) | [열기](nlp_llm/solutions/07_gpt3_few_shot.ipynb) |
| 08 | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021) | frozen W0 + BA low-rank update와 trainable parameter 절감을 재현한다. | `nlp_corpus.json` | [열기](nlp_llm/exercises/08_lora.ipynb) | [열기](nlp_llm/solutions/08_lora.ipynb) |
| 09 | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (2020) | retriever distribution과 RAG-Sequence marginalization을 재현한다. | `nlp_corpus.json` | [열기](nlp_llm/exercises/09_rag.ipynb) | [열기](nlp_llm/solutions/09_rag.ipynb) |

## 생성 모델

| 번호 | 논문 (연도) | 핵심 미니 재현 | 데이터 | 실습 | 정답 |
|---:|---|---|---|---|---|
| 00 | [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://www.jmlr.org/papers/v11/vincent10a.html) (2010) | 8×8 로컬 영상을 훼손한 뒤 깨끗한 입력을 복원하는 denoising autoencoder를 학습한다. | `generative_samples.npz` | [열기](generative/exercises/00_stacked_denoising_autoencoder.ipynb) | [열기](generative/solutions/00_stacked_denoising_autoencoder.ipynb) |
| 01 | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (2014) | 2차원 로컬 점에서 reparameterization, 닫힌형 KL, SGVB ELBO를 구현한다. | `generative_samples.npz` | [열기](generative/exercises/01_vae.ipynb) | [열기](generative/solutions/01_vae.ipynb) |
| 02 | [Generative Adversarial Nets](https://papers.nips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html) (2014) | 2차원 점 분포에서 discriminator와 non-saturating generator를 번갈아 학습한다. | `generative_samples.npz` | [열기](generative/exercises/02_gan.ipynb) | [열기](generative/solutions/02_gan.ipynb) |
| 03 | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) (2016) | 8×8 영상용 작은 DCGAN에서 strided convolution, batch normalization, tanh 설계를 검증한다. | `generative_samples.npz` | [열기](generative/exercises/03_dcgan.ipynb) | [열기](generative/solutions/03_dcgan.ipynb) |
| 04 | [Wasserstein GAN](https://proceedings.mlr.press/v70/arjovsky17a.html) (2017) | 2차원 분포에서 sigmoid 없는 critic, n_critic 업데이트, weight clipping을 구현한다. | `generative_samples.npz` | [열기](generative/exercises/04_wgan.ipynb) | [열기](generative/solutions/04_wgan.ipynb) |
| 05 | [Image-to-Image Translation with Conditional Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html) (2017) | paired 2D domains에서 conditional discriminator와 adversarial+L1 generator objective를 재현한다. | `generative_samples.npz` | [열기](generative/exercises/05_pix2pix.ipynb) | [열기](generative/solutions/05_pix2pix.ipynb) |
| 06 | [Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html) (2017) | unpaired 2D domains에서 두 generator와 두 discriminator를 adversarial+cycle loss로 학습한다. | `generative_samples.npz` | [열기](generative/exercises/06_cyclegan.ipynb) | [열기](generative/solutions/06_cyclegan.ipynb) |
| 07 | [Denoising Diffusion Probabilistic Models](https://papers.nips.cc/paper_files/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) (2020) | 8×8 로컬 영상에서 closed-form forward noising, epsilon objective, reverse sampling을 구현한다. | `generative_samples.npz` | [열기](generative/exercises/07_ddpm.ipynb) | [열기](generative/solutions/07_ddpm.ipynb) |
| 08 | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) (2021) | 2D 점에서 VP-SDE marginal score를 학습하고 reverse-time Euler sampling을 수행한다. | `generative_samples.npz` | [열기](generative/exercises/08_score_sde.ipynb) | [열기](generative/solutions/08_score_sde.ipynb) |
| 09 | [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) (2022) | 8×8 영상을 8D latent로 압축한 뒤 latent-space noise predictor와 decoder sampling을 구현한다. | `generative_samples.npz` | [열기](generative/exercises/09_latent_diffusion.ipynb) | [열기](generative/solutions/09_latent_diffusion.ipynb) |

## 강화학습·에이전트

| 번호 | 논문 (연도) | 핵심 미니 재현 | 데이터 | 실습 | 정답 |
|---:|---|---|---|---|---|
| 00 | [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696) (1992) | 로컬 bandit means에서 reward-baseline과 characteristic eligibility로 policy gradient를 재현한다. | `gridworld.json` | [열기](reinforcement_learning/exercises/00_reinforce.ipynb) | [열기](reinforcement_learning/solutions/00_reinforce.ipynb) |
| 01 | [Human-level Control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236) (2015) | 로컬 transition replay에서 one-hot DQN과 frozen target network의 TD loss를 학습한다. | `gridworld.json` | [열기](reinforcement_learning/exercises/01_dqn.ipynb) | [열기](reinforcement_learning/solutions/01_dqn.ipynb) |
| 02 | [Deep Reinforcement Learning with Double Q-learning](https://ojs.aaai.org/index.php/AAAI/article/view/10295) (2016) | online action selection과 target evaluation을 분리하고 noisy-value overestimation을 계량한다. | `gridworld.json` | [열기](reinforcement_learning/exercises/02_double_dqn.ipynb) | [열기](reinforcement_learning/solutions/02_double_dqn.ipynb) |
| 03 | [Dueling Network Architectures for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/wangf16.html) (2016) | V와 A stream을 mean-centered aggregation으로 결합해 로컬 optimal Q table을 근사한다. | `gridworld.json` | [열기](reinforcement_learning/exercises/03_dueling_dqn.ipynb) | [열기](reinforcement_learning/solutions/03_dueling_dqn.ipynb) |
| 04 | [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) (2016) | TD-error proportional sampling과 annealed importance weights로 로컬 transition Q-table을 학습한다. | `gridworld.json` | [열기](reinforcement_learning/exercises/04_prioritized_replay.ipynb) | [열기](reinforcement_learning/solutions/04_prioritized_replay.ipynb) |
| 05 | [Asynchronous Methods for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/mniha16.html) (2016) | 공유 actor-critic에 여러 seed worker rollout의 n-step advantage를 순차 적용해 A3C update를 재현한다. | `gridworld.json` | [열기](reinforcement_learning/exercises/05_a3c.ipynb) | [열기](reinforcement_learning/solutions/05_a3c.ipynb) |
| 06 | [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) (2016) | 로컬 상태에서 연속 action proxy를 만들고 deterministic actor, critic, soft target update를 학습한다. | `gridworld.json` | [열기](reinforcement_learning/exercises/06_ddpg.ipynb) | [열기](reinforcement_learning/solutions/06_ddpg.ipynb) |
| 07 | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017) | 고정 old log-prob batch에서 clipped surrogate를 여러 epoch 최적화해 로컬 optimal policy를 학습한다. | `gridworld.json` | [열기](reinforcement_learning/exercises/07_ppo.ipynb) | [열기](reinforcement_learning/solutions/07_ppo.ipynb) |
| 08 | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html) (2018) | squashed Gaussian actor와 soft actor objective를 로컬 continuous-action replay에서 학습한다. | `gridworld.json` | [열기](reinforcement_learning/exercises/08_sac.ipynb) | [열기](reinforcement_learning/solutions/08_sac.ipynb) |
| 09 | [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) (2017) | 로컬 deterministic MDP에서 PUCT visit policy와 joint policy/value loss를 구현한다. | `gridworld.json` | [열기](reinforcement_learning/exercises/09_alphazero.ipynb) | [열기](reinforcement_learning/solutions/09_alphazero.ipynb) |

## 그래프 학습 · 추천 시스템

| 번호 | 논문 (연도) | 핵심 미니 재현 | 데이터 | 실습 | 정답 |
|---:|---|---|---|---|---|
| 00 | [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) (2014) | 절단 random walk를 문장으로 보고 Skip-gram 문맥쌍을 만든 뒤 작은 그래프의 노드 임베딩을 학습합니다. | `graph_recommendation.npz` | [열기](graph_recommendation/exercises/00_deepwalk.ipynb) | [열기](graph_recommendation/solutions/00_deepwalk.ipynb) |
| 01 | [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) (2016) | p·q로 조절되는 2차 biased walk를 구현하고 BFS형/DFS형 탐색 통계를 비교합니다. | `graph_recommendation.npz` | [열기](graph_recommendation/exercises/01_node2vec.ipynb) | [열기](graph_recommendation/solutions/01_node2vec.ipynb) |
| 02 | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (2017) | renormalization trick과 2층 GCN을 구현해 일부 label만으로 노드를 분류합니다. | `graph_recommendation.npz` | [열기](graph_recommendation/exercises/02_gcn.ipynb) | [열기](graph_recommendation/solutions/02_gcn.ipynb) |
| 03 | [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (2017) | 고정 크기 이웃 표본과 mean aggregator를 구현하고 feature 기반 inductive 출력을 확인합니다. | `graph_recommendation.npz` | [열기](graph_recommendation/exercises/03_graphsage.ipynb) | [열기](graph_recommendation/solutions/03_graphsage.ipynb) |
| 04 | [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (2018) | masked self-attention 계수와 single-head GAT layer를 구현해 이웃별 가중치를 관찰합니다. | `graph_recommendation.npz` | [열기](graph_recommendation/exercises/04_gat.ipynb) | [열기](graph_recommendation/solutions/04_gat.ipynb) |
| 05 | [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) (2016) | GCN posterior encoder, reparameterization, inner-product decoder로 링크를 복원합니다. | `graph_recommendation.npz` | [열기](graph_recommendation/exercises/05_vgae.ipynb) | [열기](graph_recommendation/solutions/05_vgae.ipynb) |
| 06 | [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (2018) | 관계별 weight와 degree normalization을 갖는 R-GCN propagation을 구현합니다. | `graph_recommendation.npz` | [열기](graph_recommendation/exercises/06_rgcn.ipynb) | [열기](graph_recommendation/solutions/06_rgcn.ipynb) |
| 07 | [How Powerful are Graph Neural Networks?](https://openreview.net/forum?id=ryGs6iA5Km) (2019) | sum aggregator와 (1+epsilon) self term을 구현해 mean이 잃는 degree 정보를 보존함을 확인합니다. | `graph_recommendation.npz` | [열기](graph_recommendation/exercises/07_gin.ipynb) | [열기](graph_recommendation/solutions/07_gin.ipynb) |
| 08 | [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973) (2018) | user-item graph에서 item graph를 만들고 random-walk visit importance로 weighted convolution을 수행합니다. | `graph_recommendation.npz` | [열기](graph_recommendation/exercises/08_pinsage.ipynb) | [열기](graph_recommendation/solutions/08_pinsage.ipynb) |
| 09 | [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126) (2020) | 비선형/feature transform 없이 user-item embedding을 전파하고 BPR로 학습합니다. | `graph_recommendation.npz` | [열기](graph_recommendation/exercises/09_lightgcn.ipynb) | [열기](graph_recommendation/solutions/09_lightgcn.ipynb) |

## 자기지도 학습 · 멀티모달

| 번호 | 논문 (연도) | 핵심 미니 재현 | 데이터 | 실습 | 정답 |
|---:|---|---|---|---|---|
| 00 | [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) (2018) | encoder와 autoregressive context로 미래 latent를 batch negatives 중 식별하는 InfoNCE를 학습합니다. | `multimodal_pairs.npz` | [열기](self_supervised_multimodal/exercises/00_cpc.ipynb) | [열기](self_supervised_multimodal/solutions/00_cpc.ipynb) |
| 01 | [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) (2020) | query/key encoder, momentum update, FIFO negative queue를 갖는 MoCo 한 학습 단계를 구현합니다. | `multimodal_pairs.npz` | [열기](self_supervised_multimodal/exercises/01_moco.ipynb) | [열기](self_supervised_multimodal/solutions/01_moco.ipynb) |
| 02 | [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) (2020) | 두 augmentation view, encoder f, projection g, 대칭 NT-Xent를 구현합니다. | `multimodal_pairs.npz` | [열기](self_supervised_multimodal/exercises/02_simclr.ipynb) | [열기](self_supervised_multimodal/solutions/02_simclr.ipynb) |
| 03 | [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733) (2020) | 비대칭 online predictor와 stop-gradient target EMA로 두 view 표현을 회귀합니다. | `multimodal_pairs.npz` | [열기](self_supervised_multimodal/exercises/03_byol.ipynb) | [열기](self_supervised_multimodal/solutions/03_byol.ipynb) |
| 04 | [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) (2021) | student/teacher, sharpening, centering, EMA를 결합한 DINO self-distillation을 구현합니다. | `multimodal_pairs.npz` | [열기](self_supervised_multimodal/exercises/04_dino.ipynb) | [열기](self_supervised_multimodal/solutions/04_dino.ipynb) |
| 05 | [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (2022) | patchify, 75% random mask, visible-only encoder, lightweight decoder의 masked-pixel MSE를 구현합니다. | `multimodal_pairs.npz` | [열기](self_supervised_multimodal/exercises/05_mae.ipynb) | [열기](self_supervised_multimodal/solutions/05_mae.ipynb) |
| 06 | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021) | image/text dual encoder와 symmetric contrastive loss를 학습해 retrieval·zero-shot 분류를 수행합니다. | `multimodal_pairs.npz` | [열기](self_supervised_multimodal/exercises/06_clip.ipynb) | [열기](self_supervised_multimodal/solutions/06_clip.ipynb) |
| 07 | [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) (2021) | 일부 caption을 의도적으로 오염시키고 ALIGN의 단순 dual-encoder 대칭 normalized-softmax를 학습합니다. | `multimodal_pairs.npz` | [열기](self_supervised_multimodal/exercises/07_align.ipynb) | [열기](self_supervised_multimodal/solutions/07_align.ipynb) |
| 08 | [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) (2022) | MED의 ITC·ITM·LM 세 목적과 CapFilt의 생성/필터 단계를 작은 paired corpus에서 재현합니다. | `multimodal_pairs.npz` | [열기](self_supervised_multimodal/exercises/08_blip.ipynb) | [열기](self_supervised_multimodal/solutions/08_blip.ipynb) |
| 09 | [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (2022) | Perceiver Resampler와 tanh-gated cross-attention으로 image token을 text stream에 주입합니다. | `multimodal_pairs.npz` | [열기](self_supervised_multimodal/exercises/09_flamingo.ipynb) | [열기](self_supervised_multimodal/solutions/09_flamingo.ipynb) |

## 지식 증류 · 모델 경량화

| 번호 | 논문 (연도) | 핵심 미니 재현 | 데이터 | 실습 | 정답 |
|---:|---|---|---|---|---|
| 00 | [Distilling the Knowledge in a Neural Network](https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/) (2015) | temperature로 부드러워진 teacher 분포와 hard label을 함께 사용해 작은 student를 학습한다. | `compression_bench.npz` | [열기](distillation_compression/exercises/00_knowledge_distillation.ipynb) | [열기](distillation_compression/solutions/00_knowledge_distillation.ipynb) |
| 01 | [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550) (2015) | teacher 중간 표현을 regressor로 맞춘 뒤 출력 distillation을 수행하는 2단계 학습을 구현한다. | `compression_bench.npz` | [열기](distillation_compression/exercises/01_fitnets.ipynb) | [열기](distillation_compression/solutions/01_fitnets.ipynb) |
| 02 | [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://openreview.net/forum?id=Sks9_ajex) (2017) | CNN feature를 spatial attention map으로 요약하고 teacher와 student의 정규화된 map을 맞춘다. | `compression_bench.npz` | [열기](distillation_compression/exercises/02_attention_transfer.ipynb) | [열기](distillation_compression/solutions/02_attention_transfer.ipynb) |
| 03 | [TinyBERT: Distilling BERT for Natural Language Understanding](https://aclanthology.org/2020.findings-emnlp.372/) (2020) | Transformer student가 teacher의 hidden state, attention matrix, prediction을 함께 모방하도록 학습한다. | `compression_bench.npz` | [열기](distillation_compression/exercises/03_tinybert.ipynb) | [열기](distillation_compression/solutions/03_tinybert.ipynb) |
| 04 | [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) (2016) | magnitude pruning→weight-sharing quantization→entropy coding의 세 단계를 작은 weight tensor에 적용한다. | `compression_bench.npz` | [열기](distillation_compression/exercises/04_deep_compression.ipynb) | [열기](distillation_compression/solutions/04_deep_compression.ipynb) |
| 05 | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://openreview.net/forum?id=rJl-b3RcF7) (2019) | dense network 학습으로 mask를 찾고 surviving weight를 초기값으로 되감아 sparse subnetwork를 재학습한다. | `compression_bench.npz` | [열기](distillation_compression/exercises/05_lottery_ticket.ipynb) | [열기](distillation_compression/solutions/05_lottery_ticket.ipynb) |
| 06 | [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html) (2018) | affine quantization, integer matrix multiplication, fake-quantization 학습을 구현한다. | `compression_bench.npz` | [열기](distillation_compression/exercises/06_integer_quantization.ipynb) | [열기](distillation_compression/solutions/06_integer_quantization.ipynb) |
| 07 | [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) (2018) | depthwise convolution과 expand→filter→linear project의 inverted residual block을 구현한다. | `compression_bench.npz` | [열기](distillation_compression/exercises/07_mobilenet_v2.ipynb) | [열기](distillation_compression/solutions/07_mobilenet_v2.ipynb) |
| 08 | [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html) (2018) | channel split·가벼운 branch·concat·channel shuffle을 구현하고 FLOPs와 실제 latency를 분리해 기록한다. | `compression_bench.npz` | [열기](distillation_compression/exercises/08_shufflenet_v2.ipynb) | [열기](distillation_compression/solutions/08_shufflenet_v2.ipynb) |
| 09 | [Once-for-All: Train One Network and Specialize it for Efficient Deployment](https://openreview.net/forum?id=HylxE1HKwS) (2020) | 여러 width를 공유하는 supernet을 progressive shrinking으로 학습하고 latency budget에 맞는 subnet을 선택한다. | `compression_bench.npz` | [열기](distillation_compression/exercises/09_once_for_all.ipynb) | [열기](distillation_compression/solutions/09_once_for_all.ipynb) |

## 추천 사용 순서

1. 관심 분야 하나를 선택하고 00~09를 순서대로 진행합니다.
2. 원 논문 링크에서 mapping 표에 적힌 절·식·그림만 먼저 읽습니다.
3. 실습본 TODO를 구현하고 shape·확률·gradient·mask assertion을 통과시킵니다.
4. 정답본의 같은 셀에서 필요한 부분만 비교합니다.
5. 마지막에는 seed를 유지한 채 변수 하나만 바꾸고 metric 전후를 기록합니다.

분야를 넘나드는 추천 경로는 `Vision → 자기지도·멀티모달`, `NLP·LLM → RAG/멀티모달`, `생성모델 → 확산`, `강화학습 → 추천`, `기본 모델 → 증류·경량화`입니다.
