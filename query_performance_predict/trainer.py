import torch
from utils import save_model, calculate_errors
from tqdm import tqdm

def dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                    performance_scaler, args, save_path, writer, device):
    start_epoch = 0
    best_error = 1000000
    count = 0

    for epoch in tqdm(range(start_epoch, args.dipredict_n_epochs),
                      total=len(range(start_epoch, args.dipredict_n_epochs))):
        model.train()

        epoch_loss = 0

        for batch_index, batch_data in enumerate(dataloader):
            optimizer.zero_grad()

            batch_feature_train, batch_performance_train = batch_data

            batch_output_train = model(batch_feature_train)

            loss = model.calculate_loss(batch_output_train, batch_performance_train)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        # print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
        # writer.add_scalar('Training Epoch Loss', epoch_loss, epoch + 1)

        if epoch < args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:
            save_model(model, optimizer, epoch, save_path)

        if epoch >= args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:
            model.eval()

            with torch.no_grad():
                predicted_performances = model(feature_valid)
                predicted_performances_n = performance_scaler.inverse_transform(predicted_performances)
                predicted_performances_n[:, 1:] = torch.pow(10, predicted_performances_n[:, 1:])

                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_valid, predicted_performances)

                final_mean_error = torch.mean(mean_errors).item()
                final_mean_errors_percent = torch.mean(mean_errors_percent).item()
                final_mean_qerror = torch.mean(mean_qerrors).item()

                # print('the current prediction errors are:')
                # print(f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')

                if final_mean_error < best_error:
                    best_error = final_mean_error
                    save_model(model, optimizer, epoch, save_path)
                    count = 0

                    # print('the best prediction errors are:')
                    # print(f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')
                else:
                    count += 1

                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_valid_raw, predicted_performances_n)
                # print('the real prediction errors are:：')
                # print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}, mean_qerror:{mean_qerrors}')

            if count == args.max_count:
                break