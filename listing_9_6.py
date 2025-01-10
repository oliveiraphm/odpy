from skrub.datasets import make_deduplication_data
from skrub import deduplicate

duplicated_names = make_deduplication_data(
    examples=["online course", "seminar", "conference", "in-person class", 
              "lecture series"],  
    entries_per_example=[500, 500, 500, 500, 500],  
    prob_mistake_per_letter=0.1,  
    random_state=42,  
)
deduplicated_data = deduplicate(duplicated_names)

print(deduplicated_data)