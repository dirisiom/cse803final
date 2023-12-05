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


def train_model(m, c, o, train_l, val_l, epochs=10):
    losses = []
    v_losses = []
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
        print()
        print(f'Epoch {e + 1}/{epochs}, Loss: {e_loss}')

        m.eval()
        running_v = 0.0
        with torch.no_grad():
            for i, l in tqdm(val_l):
                i, l = i.to(device), l.to(device)
                out = m(i)
                running_v += c(out, l).item()
        e_v_loss = running_v / len(val_l)
        v_losses.append(e_v_loss)
        print(f'Epoch {e + 1}/{epochs}, Validation Loss: {e_v_loss}')

    print('Done training!')
    return losses, v_losses


def plot_results(m, loader, title):
    m.eval()
    all_preds = []
    all_labels = []
    hard_labels_list = ['A', 'E', 'I', 'J', 'N', 'M', 'S', 'T']
    hard_preds = []
    hard_labels = []
    label_to_idx = {label: idx for idx, label in enumerate(loader.dataset.dataset.classes)}
    hard_label_indices = {label_to_idx[label] for label in hard_labels_list if label in label_to_idx}
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

            for p, l in zip(pred.view(-1).cpu().numpy(), labels.view(-1).cpu().numpy()):
                if l in hard_label_indices:
                    if p in hard_label_indices:
                        hard_labels.append(l)
                        hard_preds.append(p)

        acc = 100 * correct / total
        print(f'Accuracy: {acc}')
    plot_cm(all_labels, all_preds, loader.dataset.dataset.classes, 'General', title)
    plot_cm(hard_labels, hard_preds, hard_labels_list, 'Hard', title, hard=True)


def plot_cm(true, pred, classes, title, m_title, hard=False):
    cm = confusion_matrix(true, pred)
    if not hard:
        plt.figure(figsize=(21, 21))
    else:
        plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{title} Confusion Matrix, {m_title} Data')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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
    plt.show()

def plot_loss(t_losses, v_losses):
    plt.figure(figsize=(10,6))
    plt.plot(t_losses, label='Training Loss')
    plt.plot(v_losses, label='Validation Loss')
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
        losses, v_losses = train_model(model, crit, optimizer, train_loader, val_loader)
        # Saving Model State Dictionary
        state_dict_path = 'data/asl_classifier_state_dict.pth'
        torch.save(model.state_dict(), state_dict_path)
        print(f"State dictionary saved to {state_dict_path}")
        plot_loss(losses, v_losses)

        # plot_results(model, val_loader)

    if results:
        model.load_state_dict(torch.load('data/asl_classifier_state_dict.pth'))
        model.eval()
        plot_results(model, train_loader, 'Train')
        # plot_results(model, val_loader, 'Validation')
        plot_results(model, test_loader, 'Test')


if __name__ == '__main__':
    # main()
    # main(train_p=True, results=True)
    main(results=True)
