import jsonlines
import os
import time
import csv
import openai

model = "ft:gpt-3.5-turbo-0125:personal::9DbeHFcw" # use gpt-3.5-turbo custom
# model = "gpt-3.5-turbo" # use gpt-3.5-turbo by default

def setup_api_key():
    os.environ["OPENAI_API_KEY"] = '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'

def create_fine_tuning_file(file_path):
    print("Processing fine tuning file " + file_path)
    file = openai.files.create(
        file=open(file_path, "rb"),
        purpose='fine-tune'
    )

    print(file)
    print(f'file: {file.id}')


    # Get the file ID
    file_id = file.id

    # Check the file's status
    status = file.status

    while status != 'processed':
        print(f"File status: {status}. Waiting for the file to be processed...")
        time.sleep(10)  # Wait for 10 seconds
        file_response = openai.files.retrieve(file_id)
        status = file_response.status
        print(file_response)

    fine_tuning_response = openai.fine_tuning.jobs.create(training_file=file_id, model=model)
    print(fine_tuning_response)
    return fine_tuning_response

def fine_tune_model(fine_tuning_file):
    print("Starting fine tuning job with ID: " + fine_tuning_file.id)
    if fine_tuning_file.status == 'processed':
        fine_tuning_response = openai.fine_tuning.jobs.create(
            training_file=fine_tuning_file.id,
            model=model
        )
        print(fine_tuning_response.id)

def load_csv_finetuning(csv_file, output_path, only_system=False):
    # Open the CSV file for reading
    with open(csv_file, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Open the JSONL file for writing
        with jsonlines.open(output_path, mode='w') as jsonl_file:
            for row in csv_reader:
                system = row[0]
                values = [{"role": "system", "content": system}]
                odd = True
                for value in row[1:]:
                    if odd:
                        if len(value) > 0:
                            # FORCE ONLY ASSISTANT IF only_system IS TRUE
                            role = "assistant" if only_system else "user"                         
                            values.append({"role": role, "content": value})
                        odd = False
                    else:
                        if len(value) > 0:
                            values.append({"role": "assistant", "content": value})
                        odd = True
                json_data = {"messages":values}
                jsonl_file.write(json_data)

if __name__ == '__main__':
    setup_api_key()
    fine_tuning_data = "WHERE_YOU_WANT_TO_OUTPUT.jsonl"
    load_csv_finetuning("YOUR_RAW_FINETUNING_DATA.csv", fine_tuning_data, only_system=True)
    fine_tuning_file = create_fine_tuning_file(fine_tuning_data)
    id = fine_tune_model(fine_tuning_file)
    print(openai.fine_tuning.jobs.list(limit=10))