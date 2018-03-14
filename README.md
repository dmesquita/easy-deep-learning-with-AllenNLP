# Deep Learning for text made easy with AllenNLP

AllenNLP text classification tutorial

# Run the model
(notebook)

```python
experiment_parameters = 'https://raw.githubusercontent.com/dmesquita/easy-deep-learning-with-AllenNLP/master/experiments/newsgroups_with_cuda.json'
```


```python
train_model_from_file(experiment_parameters,"/temp_dir") 
```

    1774it [00:16, 105.37it/s]
    1180it [00:12, 96.48it/s]
    2954it [00:02, 1404.15it/s]
    accuracy: 0.33, accuracy3: 0.96, loss: 1.45 ||: 100%|##########| 28/28 [00:06<00:00,  4.07it/s]
    accuracy: 0.37, accuracy3: 1.00, loss: 1.12 ||: 100%|##########| 28/28 [00:04<00:00,  5.78it/s]
    accuracy: 0.44, accuracy3: 1.00, loss: 1.05 ||: 100%|##########| 28/28 [00:04<00:00,  5.70it/s]
    accuracy: 0.54, accuracy3: 1.00, loss: 0.97 ||: 100%|##########| 28/28 [00:04<00:00,  5.76it/s]
    accuracy: 0.59, accuracy3: 1.00, loss: 0.88 ||: 100%|##########| 28/28 [00:04<00:00,  5.75it/s]
    accuracy: 0.65, accuracy3: 1.00, loss: 0.77 ||: 100%|##########| 28/28 [00:04<00:00,  5.76it/s]
    accuracy: 0.68, accuracy3: 1.00, loss: 0.74 ||: 100%|##########| 28/28 [00:04<00:00,  5.71it/s]
    accuracy: 0.75, accuracy3: 1.00, loss: 0.61 ||: 100%|##########| 28/28 [00:04<00:00,  5.72it/s]
    accuracy: 0.78, accuracy3: 1.00, loss: 0.55 ||: 100%|##########| 28/28 [00:04<00:00,  5.76it/s]
    accuracy: 0.80, accuracy3: 1.00, loss: 0.48 ||:  21%|##1       | 6/28 [00:00<00:03,  6.37it/s]accuracy: 0.77, accuracy3: 1.00, loss: 0.55 ||: 100%|##########| 28/28 [00:04<00:00,  5.71it/s]
    accuracy: 0.81, accuracy3: 1.00, loss: 0.48 ||: 100%|##########| 28/28 [00:04<00:00,  5.73it/s]
    accuracy: 0.85, accuracy3: 1.00, loss: 0.40 ||: 100%|##########| 28/28 [00:04<00:00,  5.78it/s]
    accuracy: 0.89, accuracy3: 1.00, loss: 0.30 ||: 100%|##########| 28/28 [00:04<00:00,  5.72it/s]
    accuracy: 0.90, accuracy3: 1.00, loss: 0.28 ||: 100%|##########| 28/28 [00:04<00:00,  5.78it/s]
    accuracy: 0.91, accuracy3: 1.00, loss: 0.25 ||: 100%|##########| 28/28 [00:04<00:00,  5.75it/s]
    accuracy: 0.90, accuracy3: 1.00, loss: 0.27 ||: 100%|##########| 28/28 [00:04<00:00,  5.79it/s]
    accuracy: 0.94, accuracy3: 1.00, loss: 0.16 ||: 100%|##########| 28/28 [00:04<00:00,  5.76it/s]
    accuracy: 0.96, accuracy3: 1.00, loss: 0.13 ||: 100%|##########| 28/28 [00:04<00:00,  5.73it/s]
    accuracy: 0.97, accuracy3: 1.00, loss: 0.09 ||: 100%|##########| 28/28 [00:05<00:00,  5.47it/s]
    accuracy: 0.97, accuracy3: 1.00, loss: 0.10 ||:  43%|####2     | 12/28 [00:03<00:04,  3.55it/s]accuracy: 0.97, accuracy3: 1.00, loss: 0.09 ||: 100%|##########| 28/28 [00:04<00:00,  5.74it/s]
    accuracy: 0.98, accuracy3: 1.00, loss: 0.07 ||: 100%|##########| 28/28 [00:04<00:00,  5.77it/s]
    accuracy: 0.99, accuracy3: 1.00, loss: 0.04 ||: 100%|##########| 28/28 [00:04<00:00,  5.72it/s]
    accuracy: 0.99, accuracy3: 1.00, loss: 0.04 ||: 100%|##########| 28/28 [00:04<00:00,  5.78it/s]
    accuracy: 0.99, accuracy3: 1.00, loss: 0.03 ||: 100%|##########| 28/28 [00:04<00:00,  5.73it/s]
    accuracy: 1.00, accuracy3: 1.00, loss: 0.02 ||: 100%|##########| 28/28 [00:04<00:00,  5.78it/s]
    accuracy: 1.00, accuracy3: 1.00, loss: 0.01 ||: 100%|##########| 28/28 [00:04<00:00,  5.77it/s]
    accuracy: 1.00, accuracy3: 1.00, loss: 0.01 ||: 100%|##########| 28/28 [00:04<00:00,  5.79it/s]
    accuracy: 1.00, accuracy3: 1.00, loss: 0.02 ||: 100%|##########| 28/28 [00:04<00:00,  5.73it/s]
    accuracy: 1.00, accuracy3: 1.00, loss: 0.01 ||: 100%|##########| 28/28 [00:04<00:00,  5.69it/s]
    accuracy: 1.00, accuracy3: 1.00, loss: 0.01 ||:  64%|######4   | 18/28 [00:02<00:01,  8.77it/s]accuracy: 1.00, accuracy3: 1.00, loss: 0.01 ||: 100%|##########| 28/28 [00:04<00:00,  5.72it/s]
    accuracy: 0.84, accuracy3: 1.00 ||: 100%|##########| 19/19 [00:03<00:00,  5.91it/s]




