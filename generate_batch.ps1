$prompts = @(
    "A bioluminescent jellyfish gracefully floating in the deep dark ocean, highly detailed",
    "A stack of ancient, magical books on a dusty wooden table, enchanted library",
    "A whimsical treehouse village built on giant, glowing mushrooms, fantasy concept art"
)

$output_names = @(
    "jellyfish.png",
    "magic_books.png",
    "mushroom_village.png"
)

for ($i = 0; $i -lt $prompts.Length; $i++) {
    $current_prompt = $prompts[$i]
    $current_output = "outputs\" + $output_names[$i] # 確保 outputs 資料夾存在
    Write-Host "Generating: $current_prompt -> $current_output"
    & "C:\Users\Admin\Desktop\Masters_Programs\Machine Learning\Python\ai_image_generation_project\venv\Scripts\python.exe" src\main.py --demo diffusion --prompt "$current_prompt" --output_path "$current_output" --num_steps 50
}

Write-Host "All images generated!"