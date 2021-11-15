<h1 align="center">
  Meta Deformable DETR based Metric learning</br>
  For Few-shot object ddetection
</h1>
<p align="center">Official PyTorch implementation of SKT AI Fellowship 3 <b>Meta Drformable DETR based Metric learning</b></p>
<p align="center"><img src="paper-01.png" alt="graph" width="55%"></p>

<p align="center"><a href="https://github.com/create-go-app/cli/releases" target="_blank"><img src="https://img.shields.io/badge/version-v3.2.1-blue?style=for-the-badge&logo=none" alt="cli version" /></a>&nbsp;<a href="https://pkg.go.dev/github.com/create-go-app/cli/v3?tab=doc" target="_blank"><img src="https://img.shields.io/badge/Go-1.17+-00ADD8?style=for-the-badge&logo=go" alt="go version" /></a>&nbsp;<a href="https://gocover.io/github.com/create-go-app/cli/pkg/cgapp" target="_blank"><img src="https://img.shields.io/badge/Go_Cover-89.2%25-success?style=for-the-badge&logo=none" alt="go cover" /></a>&nbsp;<a href="https://goreportcard.com/report/github.com/create-go-app/cli" target="_blank"><img src="https://img.shields.io/badge/Go_report-A+-success?style=for-the-badge&logo=none" alt="go report" /></a>&nbsp;<img src="https://img.shields.io/badge/license-apache_2.0-red?style=for-the-badge&logo=none" alt="license" /></p>

## ⚡️ Quick start
### Requirements
```bash
pip install -r requirements.txt
```

### Datasets

1. Download four public benchmarks for deep metric learning
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))
   - In-shop Clothes Retrieval ([Link](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html))

2. Extract the tgz or zip file into `./data/` (Exceptionally, for Cars-196, put the files in a `./data/cars196`)

**[Notice!]** I found that the link that was previously uploaded for the CUB dataset was incorrect, so I corrected the link. (CUB-200 -> CUB-200-2011)
If you have previously downloaded the CUB dataset from my repository, please download it again. 
Thanks to myeongjun for reporting this issue!

### Training Embedding Network

Note that a sufficiently large batch size and good parameters resulted in better overall performance than that described in the paper. You can download the trained model through the hyperlink in the table.

### CUB-200-2011

- Train a embedding network of Inception-BN (d=512) using **Proxy-Anchor loss**

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10
```

- Train a embedding network of ResNet-50 (d=512) using **Proxy-Anchor loss**

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model resnet50 \
                --embedding-size 512 \
                --batch-size 120 \
                --lr 1e-4 \
                --dataset cub \
                --warm 5 \
                --bn-freeze 1 \
                --lr-decay-step 5
```

| Method | Backbone | R@1 | R@2 | R@4 | R@8 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1twaY6S2QIR8eanjDB6PoVPlCTsn-6ZJW/view?usp=sharing) | Inception-BN | 69.1 | 78.9 | 86.1 | 91.2 |
| [Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1s-cRSEL2PhPFL9S7bavkrD_c59bJXL_u/view?usp=sharing) | ResNet-50 | 69.9 | 79.6 | 86.6 | 91.4 |

### Cars-196

- Train a embedding network of Inception-BN (d=512) using **Proxy-Anchor loss**

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20
```

- Train a embedding network of ResNet-50 (d=512) using **Proxy-Anchor loss**

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model resnet50 \
                --embedding-size 512 \
                --batch-size 120 \
                --lr 1e-4 \
                --dataset cars \
                --warm 5 \
                --bn-freeze 1 \
                --lr-decay-step 10 
```

| Method | Backbone | R@1 | R@2 | R@4 | R@8 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1wwN4ojmOCEAOaSYQHArzJbNdJQNvo4E1/view?usp=sharing) | Inception-BN | 86.4 | 91.9 | 95.0 | 97.0 |
| [Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1_4P90jZcDr0xolRduNpgJ9tX9HZ1Ih7n/view?usp=sharing) | ResNet-50 | 87.7 | 92.7 | 95.5 | 97.3 |

### Stanford Online Products

- Train a embedding network of Inception-BN (d=512) using **Proxy-Anchor loss**

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 0 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25
```

| Method | Backbone | R@1 | R@10 | R@100 | R@1000 |
|:-:|:-:|:-:|:-:|:-:|:-:|
|[Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1hBdWhLP2J83JlOMRgZ4LLZY45L-9Gj2X/view?usp=sharing) | Inception-BN | 79.2 | 90.7 | 96.2 | 98.6 |

### In-Shop Clothes Retrieval

- Train a embedding network of Inception-BN (d=512) using **Proxy-Anchor loss**

```bash
python train.py --gpu-id 0 \
                --loss Proxy_Anchor \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 6e-4 \
                --dataset Inshop \
                --warm 1 \
                --bn-freeze 0 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25
```

| Method | Backbone | R@1 | R@10 | R@20 | R@30 | R@40 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1VE7psay7dblDyod8di72Sv7Z2xGtUGra/view?usp=sharing) | Inception-BN | 91.9 | 98.1 | 98.7 | 99.0 | 99.1 |



## Evaluating Image Retrieval

Follow the below steps to evaluate the provided pretrained model or your trained model. 

Trained best model will be saved in the `./logs/folder_name`.

```bash
# The parameters should be changed according to the model to be evaluated.
python evaluate.py --gpu-id 0 \
                   --batch-size 120 \
                   --model bn_inception \
                   --embedding-size 512 \
                   --dataset cub \
                   --resume /set/your/model/path/best_model.pth
```



First of all, [download](https://golang.org/dl/) and install **Go**. Version `1.17` or higher is required.

> If you're looking for the **Create Go App CLI** for Go `1.16`, you can find it [here](https://github.com/create-go-app/cli/tree/v2).

Installation is done by using the [`go install`](https://golang.org/cmd/go/#hdr-Compile_and_install_packages_and_dependencies) command and rename installed binary in `$GOPATH/bin`:

```bash
go install github.com/create-go-app/cli/v3/cmd/cgapp@latest
```

Also, macOS and GNU/Linux users available way to install via [Homebrew](https://brew.sh/):

```bash
# Tap a new formula:
brew tap create-go-app/cli

# Installation:
brew install create-go-app/cli/cgapp
```

Let's create a new project via **interactive console UI** (or **CUI** for short) in current folder:

```bash
cgapp create
```

Next, open the generated Ansible inventory file (called `hosts.ini`) and fill in the variables according to your server configuration. And you're ready to **automatically deploy** this project:

```bash
cgapp deploy
```

That's all you need to know to start! 🎉

### 🐳 Docker-way to quick start

If you don't want to install Create Go App CLI to your system, you feel free to using our official [Docker image](https://hub.docker.com/r/koddr/cgapp) and run CLI from isolated container:

```bash
docker run --rm -it -v ${PWD}:${PWD} -w ${PWD} koddr/cgapp:latest [COMMAND]
```

> 🔔 Please note: the `deploy` command is currently **unavailable** in this image.

## 📖 Project Wiki

The best way to better explore all the features of the **Create Go App CLI** is to read the project [Wiki](https://github.com/create-go-app/cli/wiki) and take part in [Discussions](https://github.com/create-go-app/cli/discussions) and/or [Issues](https://github.com/create-go-app/cli/issues). Yes, the most frequently asked questions (_FAQ_) are also [here](https://github.com/create-go-app/cli/wiki/FAQ).

## ⚙️ Commands & Options

### `create`

CLI command for create a new project with the interactive console UI.

```bash
cgapp create [OPTION]
```

| Option | Description                                              | Type   | Default | Required? |
| ------ | -------------------------------------------------------- | ------ | ------- | --------- |
| `-t`   | Enables to define custom backend and frontend templates. | `bool` | `false` | No        |

![cgapp_create](https://user-images.githubusercontent.com/11155743/116796937-38160080-aae9-11eb-8e21-fb1be2750aa4.gif)

- 📺 Full demo video: https://recordit.co/OQAwkZBrjN
- 📖 Docs: https://github.com/create-go-app/cli/wiki/Command-create

### `deploy`

CLI command for deploy Docker containers with your project via Ansible to the remote server.

> 🔔 Make sure that you have [Python 3.8+](https://www.python.org/downloads/) and [Ansible 2.9+](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html#installing-ansible-on-specific-operating-systems) installed on your computer.

```bash
cgapp deploy [OPTION]
```

| Option | Description                                                                                            | Type   | Default | Required? |
| ------ | ------------------------------------------------------------------------------------------------------ | ------ | ------- | --------- |
| `-k`   | Prompt you to provide the remote user sudo password (_a standard Ansible `--ask-become-pass` option_). | `bool` | `false` | No        |

![cgapp_deploy](https://user-images.githubusercontent.com/11155743/116796941-3c421e00-aae9-11eb-9575-d72550814d7a.gif)

- 📺 Full demo video: https://recordit.co/ishTf0Au1x
- 📖 Docs: https://github.com/create-go-app/cli/wiki/Command-deploy

## 📝 Production-ready project templates

### Backend

- Backend template with Golang built-in [net/http](https://golang.org/pkg/net/http/) package:
  - [`net/http`](https://github.com/create-go-app/net_http-go-template) — simple REST API with CRUD and JWT auth.
- Backend template with [Fiber](https://github.com/gofiber/fiber):
  - [`fiber`](https://github.com/create-go-app/fiber-go-template) — complex REST API with CRUD, JWT auth with renew token, DB and cache.

### Frontend

- Pure JavaScript frontend template:
  - `vanilla` — generated template with pure JavaScript app.
  - `vanilla-ts` — generated template with pure TypeScript app.
- Frontend template with [React](https://reactjs.org/):
  - `react` — generated template with a common React app.
  - `react-ts` — generated template with a TypeScript version of the React app.
- Frontend template with [Preact](https://preactjs.com/):
  - `preact` — generated template with a common Preact app.
  - `preact-ts` — generated template with a TypeScript version of the Preact app.
- Frontend template with [Vue.js](https://vuejs.org/):
  - `vue` — generated template with a common Vue.js app.
  - `vue-ts` — generated template with a TypeScript version of the Vue.js app.
- Frontend template with [Svelte](https://svelte.dev/):
  - `svelte` — generated template with a common Svelte app.
  - `svelte-ts` — generated template with a TypeScript version of the Svelte app.
- Frontend template with [Lit](https://lit.dev/) web components:
  - `lit-element` — generated template with a common Lit app.
  - `lit-element-ts` — generated template a TypeScript version of the Lit app.

> ☝️ Frontend part will be generate using awesome tool [Vite.js](https://vitejs.dev/) under the hood. So, you'll always get the latest version of `React`, `Preact`, `Vue`, `Svelte`, `Lit` or pure JavaScript/TypeScript templates for your project!
>
> Please make sure that you have `npm` version `7` or higher installed to create the frontend part of the project correctly. If you run the `cgapp create` command using our [Docker image](https://hub.docker.com/r/koddr/cgapp), `npm` of the correct version is **already** included.

## 🚚 Pre-configured Ansible roles

### Web/Proxy server

- Roles for run Docker container with [Traefik Proxy](https://traefik.io/traefik/):
  - `traefik` — configured Traefik container with a simple ACME challenge via CA server.
  - `traefik-acme-dns` — configured Traefik container with a complex ACME challenge via DNS provider.
- Roles for run Docker container with [Nginx](https://nginx.org):
  - `nginx` — pure Nginx container with "the best practice" configuration.

> ✌️ Since Create Go App CLI `v2.0.0`, we're recommend to use **Traefik Proxy** as default proxy server for your projects. The main reason: this proxy provides _automatic_ SSL certificates from Let's Encrypt out of the box. Also, Traefik was built on the Docker ecosystem and has a _really good looking_ and _useful_ Web UI.

### Database

- Roles for run Docker container with [PostgreSQL](https://postgresql.org/):
  - `postgres` — configured PostgreSQL container with apply migrations for backend.

### Cache (key-value storage)

- Roles for run Docker container with [Redis](https://redis.io/):
  - `redis` — configured Redis container for backend.

## ⭐️ Project assistance

If you want to say **thank you** or/and support active development of `Create Go App CLI`:

- Add a [GitHub Star](https://github.com/create-go-app/cli) to the project.
- Tweet about project [on your Twitter](https://twitter.com/intent/tweet?text=%E2%9C%A8%20Create%20a%20new%20production-ready%20project%20with%20%23Golang%20backend%2C%20%23JavaScript%20or%20%23TypeScript%20frontend%2C%20%23Docker%20and%20%23Ansible%20deploy%20automation%20by%20running%20one%20command.%20%0A%0AFocus%20on%20writing%20code%20and%20thinking%20of%20business-logic%21%0AThe%20CLI%20will%20take%20care%20of%20the%20rest.%0A%0Ahttps%3A%2F%2Fgithub.com%2Fcreate-go-app%2Fcli).
- Write interesting articles about project on [Dev.to](https://dev.to/), [Medium](https://medium.com/) or personal blog.
- Join DigitalOcean at our [referral link](https://m.do.co/c/b41859fa9b6e) (your profit is **$100** and we get $25).

<a href="https://www.producthunt.com/posts/create-go-app?utm_source=badge-review&utm_medium=badge&utm_souce=badge-create-go-app#discussion-body" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/review.svg?post_id=316086&theme=light" alt="Create Go App - Create a new production-ready project by one CLI command | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>

Together, we can make this project **better** every day! 😘

## ⚠️ License

`Create Go App CLI` is free and open-source software licensed under the [Apache 2.0 License](https://github.com/create-go-app/cli/blob/master/LICENSE). Official [logo](https://github.com/create-go-app/cli/wiki/Logo) was created by [Vic Shóstak](https://shostak.dev/) and distributed under [Creative Commons](https://creativecommons.org/licenses/by-sa/4.0/) license (CC BY-SA 4.0 International).
T
