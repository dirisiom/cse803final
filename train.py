import itertools

from data import *
from models import *
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'Using {device}')


def train_model(m, c, o, train_l, epochs=10):
    for e in range(epochs):
        m.train()
        running = 0.0
        print('inner loop now')
        counter = 0
        for inputs, labels in train_l:
            inputs, labels = inputs.to(device), labels.to(device)
            o.zero_grad()
            out = m(inputs)
            loss = c(out, labels)
            loss.backward()
            o.step()
            running += loss
            if counter % 25 == 0:
                print(counter)
            counter += 1
        print(f'Epoch {e+1}/{epochs}, Loss: {running / len(train_l)}')
    print('Done training!')


def plot_results(m, loader):
    m.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = m(images)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        acc = 100 * correct / total
        print(f'Accuracy: {acc}')
        all_preds.extend(pred.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(loader.dataset.dataset.classes))
    plt.xticks(tick_marks, loader.dataset.dataset.classes, rotation=45)
    plt.yticks(tick_marks, loader.dataset.dataset.classes)

    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def main(train=False):
    data = fetch_data()

    # Split dataset
    train_size = int(0.7 * len(data))
    val_size = int(.15 * len(data))
    test_size = len(data) - (train_size + val_size)
    train, val, test = random_split(data, [train_size, val_size, test_size])

    # Data Loaders
    batch_size = 32
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    # TODO: this will likely need to change in order to accommodate different model types
    model = ASLCNN(len(data.classes)).to(device)

    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=.001)


    print('start training')
    train_model(model, crit, optimizer, train_loader)
    plot_results(model, val_loader)

    # Saving Model State Dictionary
    state_dict_path = 'data/asl_classifier_state_dict.pth'
    torch.save(model.state_dict(), state_dict_path)
    print(f"State dictionary saved to {state_dict_path}")


if __name__ == '__main__':
    main()
