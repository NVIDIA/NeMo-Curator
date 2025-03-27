import logging
import matplotlib.pyplot as plt

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

        # Find the current stage's index and get its size
        current_index = next((i for i, (name, _) in enumerate(self.data_list) if name == stage_name), -1)
        current_size = self.data[stage_name]

        # Get previous stage's size, or use original size if this is the first stage
        previous_size = self.original_size
        if current_index > 0:
            previous_size = self.data_list[current_index - 1][1]

        filtered_out = previous_size - current_size

        if filtered_out != 0:
            # Calculate percentage based on original size
            percent_filtered = (filtered_out / self.original_size) * 100
        else:
            percent_filtered = 0.0

        self.logger.debug(f"Incremental change for {stage_name}: {filtered_out}")
        return filtered_out, percent_filtered

    def calculate_overall_change(self):
        if not self.data:
            self.logger.warning("No data recorded yet.")
            return 0

        final_stage = self.data_list[-1][0]
        final_size = self.data[final_stage]
        total_filtered = self.original_size - final_size

        if self.original_size != 0:
            # Calculate what percentage of original data was filtered out
            percent_filtered = (total_filtered / self.original_size) * 100
        else:
            percent_filtered = 0.0

        self.logger.debug(f"Overall filtered amount: {total_filtered}")
        return total_filtered, percent_filtered

    def print_summary(self):
        print("Data Processing Summary:")
        print(f"Original Size: {self.original_size}")
        print(f"Final Size: {self.data_list[-1][1]}")
        total_filtered, total_percent = self.calculate_overall_change()
        print(f"Total Filtered: {total_filtered} ({total_percent:.2f}% of original)")

        print("\nStage-wise Filtering:")
        for stage, size in self.data.items():
            filtered_amount, percent = self.calculate_incremental_change(stage)
            if filtered_amount is not None:
                print(f"  Stage: {stage}, Result Size: {size}, Filtered Out: {filtered_amount} ({percent:.2f}% of original)")
            else:
                print(f"  Stage: {stage}, Size: {size}, Filtered Amount: Not Available")

    def plot_size_reduction(self):
        """Plot the dataset size reduction after each filtering stage."""
        stages = ['Original'] + [stage for stage, _ in self.data_list]
        sizes = [self.original_size] + [size for _, size in self.data_list]

        # Calculate filtered amounts and percentages using calculate_incremental_change
        filtered_info = []
        filtered_info.append((0, 0.0))  # For original stage
        for stage, _ in self.data_list:
            filtered_amount, percent = self.calculate_incremental_change(stage)
            filtered_info.append((filtered_amount, percent))

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(range(len(stages)), sizes, marker='o')

        # Rotate x-axis labels for better readability
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=45, ha='right')

        ax.set_title('Dataset Size Reduction by Filtering Stage')
        ax.set_xlabel('Filtering Stage')
        ax.set_ylabel('Dataset Size')

        # Add both size and filtered percentage labels on top of each point
        for i, (size, (filtered_amt, pct)) in enumerate(zip(sizes, filtered_info)):
            label = f'{size:,}\n(-{filtered_amt:,}, {pct:.1f}%)' if i > 0 else f'{size:,}'
            ax.text(i, size, label, ha='center', va='bottom')

        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.show()