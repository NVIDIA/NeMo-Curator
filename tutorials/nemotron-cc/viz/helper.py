import logging

class DataSizeTracker:
    def __init__(self, original_size):
        self.original_size = original_size
        self.data = {}
        self.data_list = []  # Store data in a list to track order
        self.logger = logging.getLogger(__name__)

    def record_size(self, stage_name, size=0):
        if size == -1:
            if self.data_list:
                size = self.data_list[-1][1]
            else:
                size = self.original_size
        self.data[stage_name] = size
        self.data_list.append((stage_name, size))  # Append to the list
        self.logger.debug(f"Recorded size for {stage_name}: {size}")

    def get_data(self):
        return self.data

    def calculate_incremental_change(self, stage_name):
        if stage_name not in self.data:
            self.logger.warning(f"Stage {stage_name} not found in data.")
            return None

        # Find the index of the current stage
        current_index = -1
        for i, (name, _) in enumerate(self.data_list):
            if name == stage_name:
                current_index = i
                break

        if current_index <= 0:
            # If it's the first stage, compare to original size
            previous_size = self.original_size
        else:
            # Get the size of the previous stage
            previous_size = self.data_list[current_index - 1][1]

        current_size = self.data[stage_name]
        change = previous_size - current_size

        if previous_size != 0:
            percent_change = (change / previous_size) * 100
        else:
            percent_change = 0.0

        self.logger.debug(f"Incremental change for {stage_name}: {change}")
        return change, percent_change

    def calculate_overall_change(self):
        if not self.data:
            self.logger.warning("No data recorded yet.")
            return 0

        final_stage = self.data_list[-1][0]
        final_size = self.data[final_stage]
        overall_reduction = self.original_size - final_size

        if self.original_size != 0:
            percent_reduction = (overall_reduction / self.original_size) * 100
        else:
            percent_reduction = 0.0

        self.logger.debug(f"Overall reduction: {overall_reduction}")
        return overall_reduction, percent_reduction

    def print_summary(self):
        print("Data Processing Summary:")
        print(f"Original Size: {self.original_size}")

        overall_change, overall_percent_change = self.calculate_overall_change()
        print(f"Overall Reduction: {overall_change} ({overall_percent_change:.2f}%)")

        print("\nStage-wise Changes:")
        for stage, size in self.data.items():
            incremental_change, percent_change = self.calculate_incremental_change(stage)
            if incremental_change is not None:
                print(f"  {stage}: {size}, Incremental Change: {incremental_change} ({percent_change:.2f}%)")
            else:
                print(f"  {stage}: {size}, Incremental Change: Not Available")

        final_stage = self.data_list[-1][0]
        final_size = self.data[final_stage]
        print(f"\nFinal Size: {final_size}")
        print(f"Original Size - Overall Reduction: {self.original_size - overall_change}")