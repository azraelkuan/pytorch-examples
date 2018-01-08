# -*- coding: utf-8 -*-
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
from model import EncoderRNN, DecoderRNN, AttentionDecoderRNN
from data_feeder import prepare_data, variable_from_pair, MAX_LENGTH, SOS_token, EOS_token, variable_from_sentence
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

# parameters
use_cuda = torch.cuda.is_available()
hidden_size = 256
num_iters = 80000
print_every = 5000
plot_every = 1000
learning_rate = 0.01
teacher_forcing_ratio = 0.5
use_attention = True


def train(input_variable, output_variable, encoder, decoder, encoder_optimizer, decode_optimizer, criterion):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decode_optimizer.zero_grad()

    input_length = input_variable.size(0)
    output_length = output_variable.size(0)

    encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0.

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = False  # if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(output_length):
            if use_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.view(1, -1), output_variable[di])
            decoder_input = output_variable[di]
    else:
        for di in range(output_length):
            if use_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            top_value, top_index = decoder_output.data.topk(1)
            next_index = top_index[0][0][0]
            decoder_input = Variable(torch.LongTensor([[next_index]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output.view(1, -1), output_variable[di])

            if next_index == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decode_optimizer.step()

    return loss.data[0] / output_length


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence, use_cuda)
    input_length = input_variable.size(0)

    encoder_hidden = encoder.init_hidden()

    encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoder_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    output_length = 0

    for di in range(max_length):
        if use_attention:
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.cpu().data
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        top_value, top_index = decoder_output.data.topk(1)
        next_index = top_index[0][0][0]
        if next_index == EOS_token:
            decoder_words.append('<EOS>')
            output_length = di
            break
        else:
            decoder_words.append(output_lang.index2word[next_index])
        decoder_input = Variable(torch.LongTensor([[next_index]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoder_words, decoder_attentions[:output_length + 1]


def evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')



def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


def main():
    # data
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', reverse=True)

    encoder = EncoderRNN(input_lang.n_words, hidden_size, use_cuda)
    if use_attention:
        decoder = AttentionDecoderRNN(hidden_size, output_lang.n_words, use_cuda)
    else:
        decoder = DecoderRNN(hidden_size, output_lang.n_words, use_cuda)

    if use_cuda:
        encoder, decoder = encoder.cuda(), decoder.cuda()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    plot_losses = []
    print_total_loss = 0.
    plot_total_loss = 0.

    criterion = nn.CrossEntropyLoss()
    encoder_schedule = MultiStepLR(encoder_optimizer, [40000, 60000])
    decoder_schedule = MultiStepLR(decoder_optimizer, [40000, 60000])

    for iter in tqdm(range(1, num_iters + 1)):
        encoder_schedule.step()
        decoder_schedule.step()
        input_variable, output_variable = variable_from_pair(input_lang, output_lang, random.choice(pairs), use_cuda)
        loss = train(input_variable, output_variable, encoder, decoder, encoder_optimizer,
                     decoder_optimizer, criterion)
        print_total_loss += loss
        plot_total_loss += loss

        if iter % print_every == 0:
            print_avg_loss = print_total_loss / print_every
            print_total_loss = 0
            tqdm.write("iter: {} Percent: {}% Loss: {}".format(iter, round(100 *iter / num_iters, 2), print_avg_loss))

        if iter % plot_every == 0:
            plot_avg_loss = plot_total_loss / plot_every
            plot_losses.append(plot_avg_loss)
            plot_total_loss = 0

    show_plot(plot_losses)
    evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs)


if __name__ == '__main__':
    main()



