# from tabnanny import verbose
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from tqdm import tqdm
from torch._C import DeviceObjType
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = list(df.label)
        self.texts = [tokenizer(text, padding='max_length', max_length=128,
                                truncation=True, return_tensors="pt") for text in df.fixed_text]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(torch.nn.Module):

    def __init__(self, dropout=0.2):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = torch.nn.Dropout(dropout)
        # self.do_1 = torch.nn.Dropout(p=0.2)
        # self.do_2 = torch.nn.Dropout(p=0.2)
        # self.fc_1 = torch.nn.Linear(768, 768)
        # self.fc_2 = torch.nn.Linear(768, 2)
        self.linear = torch.nn.Linear(768, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_id, mask):

        _, out = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        # out = self.do_1(out)
        # out = self.fc_1(out)
        # out = F.relu(out)
        # out = self.do_2(out)
        # out = self.fc_2(out)
        out = self.linear(out)
        out = self.sigmoid(out)
        return out


def train(model, optimizer, train_dataloader, val_dataloader, epochs, batch_size, scheduler=None):
    criterion = torch.nn.CrossEntropyLoss()
    
    iter_train_loss = []
    iter_train_acc = []
    iter_val_loss = []
    iter_val_acc = []

    for epoch_num in range(epochs):

        model.train()

        total_acc_train = 0.
        total_loss_train = 0.

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.type(torch.LongTensor)
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            iter_train_loss.append(batch_loss.item())

            # x = (output.argmax(dim=1) == train_label).sum()
            # print(f'{output=}')
            # print(f'{train_label=}')

            acc = (output.argmax(dim=1) == train_label).sum().item() # num_of_correct
            
            # iter_train_acc.append(acc / output.shape[0])
            iter_train_acc.append(acc)
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        # load
        # model = torch.load('model_backup/best_model_FPN_Last.pth', map_location=DeviceObjType)

        # Evaluation

        # model.eval()

        for val_input, val_label in val_dataloader:
            val_label = val_label.type(torch.LongTensor)
            val_label = val_label.to(device)
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)
            # predict
            output = model(input_id, mask)
            # #tensor to numpy
            # numpy_new = your_tensor.cpu().detach().numpy()
            batch_loss = criterion(output, val_label)
            total_loss_val += batch_loss.item()
            iter_val_loss.append(batch_loss.item())

            acc = (torch.argmax(output, dim=1) == val_label).sum().item()
            # iter_val_acc.append(acc / output.shape[0])
            iter_val_acc.append(acc)
            total_acc_val += acc

        # total_acc_train /= (len(train_dataloader) * batch_size)
        # print(
        #     f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader.dataset): .6f} \
        #         | Train Accuracy: {total_acc_train: .3f} \
        #         | Val Loss: {total_loss_val / (len(val_dataloader) * batch_size): .3f} \
        #         | Val Accuracy: {total_acc_val / (len(val_dataloader) * batch_size): .3f}')

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / (len(train_dataloader) * batch_size): .6f} \
                | Train Accuracy: {total_acc_train / (len(train_dataloader) * batch_size): .3f} \
                | Val Loss: {total_loss_val / (len(val_dataloader) * batch_size): .3f} \
                | Val Accuracy: {total_acc_val / (len(val_dataloader) * batch_size): .3f}')


        torch.save(model, f'epoch{epoch_num + 1}.pth')

        # if total_acc_train > 0.99:
        #     break

        if total_loss_train / (len(train_dataloader) * batch_size) < 0.001:
            break

    torch.save(model, 'final.pth')

    train_res = pd.DataFrame(zip(iter_train_loss, iter_train_acc),
                          columns=['train_loss', 'train_acc'])

    train_res.to_csv('iter_train.csv')

    val_res = pd.DataFrame(zip(iter_val_loss, iter_val_acc),
                          columns=['val_loss', 'val_acc'])

    val_res.to_csv('iter_val.csv')

    return iter_train_loss, iter_val_loss


def test(model, test_dataloader, batch_size):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_acc_test = 0
    total_loss_test = 0
    predicted = []

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.type(torch.LongTensor)
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            # predict
            output = model(input_id, mask)
            # #tensor to numpy
            numpy_new = output.cpu().detach().numpy()
            predicted = predicted + numpy_new.tolist()
            batch_loss = criterion(output, test_label)
            total_loss_test += batch_loss.item()

            acc = (torch.argmax(output, dim=1) == test_label).sum().item()
            total_acc_test += acc

        print(
            f'Test Loss: {total_loss_test / (len(test_dataloader) * batch_size): .3f} \
                | Test Accuracy: {total_acc_test / (len(test_dataloader) * batch_size): .3f}')

    # pred = test(model, test_dataloader, BATCH_SIZE)
    pred_df = pd.DataFrame(predicted)
    pred_df['pred_label'] = [np.argmax(i) for i in predicted]
    test = pd.read_csv('NLP_test_4k.csv')
    labels = list(test.label)
    pred_df['label'] = labels
    pred_df.to_csv('NLP_test_result_e8.csv', index=False, header=True)

    return pred_df


def train_data(batch_size):
    # np.random.seed(519)
    # df = pd.read_csv('NLP_train.csv')
    # df_test = pd.read_csv('NLP_test.csv')
    # df_train, df_val = np.split(df.sample(frac=1, random_state=12), [int(.9*len(df))])
    df_train = pd.read_csv('NLP_train.csv')
    df_val = pd.read_csv('NLP_val_2k.csv')

    print(f'Training size: {len(df_train)}')
    print(f'Validation size: {len(df_val)}')

    data_train, data_val = Dataset(df_train), Dataset(df_val)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=batch_size)

    return train_dataloader, val_dataloader


def test_data(batch_size):
    df_test = pd.read_csv('NLP_test_4k.csv')
    data_test = Dataset(df_test)

    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

    return test_dataloader


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    BATCH_SIZE = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10

    # model = BertClassifier().to(device)
    # model.train()
    # optimizer = Adam(model.parameters(), lr=5e-5)
    # train_dataloader, val_dataloader = train_data(BATCH_SIZE)
    # train(model, optimizer, train_dataloader, val_dataloader, EPOCHS, BATCH_SIZE, scheduler=None)
    
    
    model = torch.load('64pad_1e-5_32BS_sigmoid/epoch8.pth').to(device)
    test_dataloader = test_data(BATCH_SIZE)
    pred_df = test(model, test_dataloader, BATCH_SIZE)


    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5, verbose=True)

    # loss_curve, val_loss_curve = train(model, optimizer, train_dataloader, val_dataloader, 5, 32, scheduler=None)

    # for param in model.bert.parameters():
    #     param.requires_grad = False

    # optimizer = Adam(model.parameters(), lr=1e-5)
    # loss_curve, val_loss_curve = train(model, optimizer, train_dataloader, val_dataloader, 5, BATCH_SIZE, scheduler=None)
    
    # plt.plot(loss_curve)
    # plt.show()
    # plt.plot(val_loss_curve)
    # plt.show()



    # model = torch.load('final.pth').to(device)
    # df_test = pd.read_csv('NLP_test.csv')
    # data_test = Dataset(df_test)
    # test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE)

    # pred = test(model, test_dataloader, BATCH_SIZE)
    # labels = list(df_test.true_label)
    # pred_df = pd.DataFrame(pred)
    # pred_df['true_label'] = labels
    # pred_df.to_csv('NLP_result.csv', index=False, header=True)



