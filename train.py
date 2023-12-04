import itertools
import os.path

import torch

from data import *
from models import *
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

generator = torch.Generator()
generator.manual_seed(5757)


def train_model(m, c, o, train_l, epochs=10):
    losses = []
    for e in range(epochs):
        m.train()
        running = 0.0
        # print('inner loop now')
        for inputs, labels in tqdm(train_l):
            inputs, labels = inputs.to(device), labels.to(device)
            o.zero_grad()
            out = m(inputs)
            loss = c(out, labels)
            loss.backward()
            o.step()
            running += loss.item()
        # Get average loss per batch
        e_loss = running / len(train_l)
        losses.append(e_loss)
        print(f'Epoch {e + 1}/{epochs}, Loss: {e_loss}')
    print('Done training!')
    return losses


def plot_results(m, loader, val=True):
    m.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            out = m(images)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())
        acc = 100 * correct / total
        print(f'Accuracy: {acc}')
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(21, 21))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    title = 'Validation Data' if val else 'Test Data'
    plt.title(f'Confusion Matrix, {title}')
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

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.subplots_adjust(left=0.19)
    plt.show()


def plot_loss(losses):
    plt.figure(figsize=(10,6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.tight_layout()
    plt.savefig('./trainlosses.png')


def main(train_p=False, results=False):
    data = fetch_data()

    # Split dataset
    train_size = int(0.7 * len(data))
    val_size = int(.15 * len(data))
    test_size = len(data) - (train_size + val_size)
    train, val, test = random_split(data, [train_size, val_size, test_size], generator=generator)

    # Data Loaders
    batch_size = 32
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    model = ASLCNN(len(data.classes)).to(device)

    if train_p or not os.path.isfile('./data/asl_classifier_state_dict.pth'):
        crit = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=.003)

        print('start training')
        losses = train_model(model, crit, optimizer, train_loader)
        # Saving Model State Dictionary
        state_dict_path = 'data/asl_classifier_state_dict.pth'
        torch.save(model.state_dict(), state_dict_path)
        print(f"State dictionary saved to {state_dict_path}")
        plot_loss(losses)

        # plot_results(model, val_loader)

    if results:
        model.load_state_dict(torch.load('data/asl_classifier_state_dict.pth'))
        model.eval()
        # plot_results(model, train_loader)
        plot_results(model, val_loader)
        plot_results(model, test_loader, val=False)


if __name__ == '__main__':
    # main()
    main(train_p=True, results=True)
    # main(results=True)
