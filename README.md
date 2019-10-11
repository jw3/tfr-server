# TFRecord Viewer
"ref: https://github.com/sulc/tfrecord-viewer"

"How about checking your data (input: *tfrecord and label_map.pbtxt) before going deeper? (updated detection only)"

Use TFRecord Viewer to browse contents of TFRecords with object detection/classification annotations.

The viewer runs a Flask server to provide a web gallery with annotation overlays.
I.e. you can run it on your server machine, but browse on your local machine.

The web gallery displayed with [Fotorama.io](https://fotorama.io/).

# Examples

`python3 tfviewer.py datasets/COCO/tfrecord/coco_train.record-0000* --label-file=label_map.pbtxt`

`python3 tfviewer.py datasets/COCO/tfrecord/coco_val.record-0000* --label-file=label_map.pbtxt`

![Detection example](http://cmp.felk.cvut.cz/~sulcmila/tfrecord-viewer/detection.png)


`python3 tfviewer.py datasets/imagenet/imagenet_fullres/tfrecord/train-00000-of-01024 --overlay classification`

![Classification example](http://cmp.felk.cvut.cz/~sulcmila/tfrecord-viewer/classification.png)
