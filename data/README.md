# Hierarchy metadata

`imagenet_hierarchy.json` is the hierarchy snapshot used by the original experiments. It was copied from the ImageNet hierarchy input supplied to the research implementation.

The JSON contains synset identifiers, names, hierarchy links, and ImageNet class indices. The loader removes the structural ImageNet container root and collapses unary nodes before measurement, using the copied hierarchy routines in `has/hierarchy.py`.

ImageNet images are not included. Obtain the training archive through the [official ImageNet download page](https://image-net.org/download.php). The hierarchy's lexical relations derive from [WordNet](https://wordnet.princeton.edu/); see the included [license](WORDNET-LICENSE) and its [license and attribution terms](https://wordnet.princeton.edu/license-and-commercial-use).
