<?php
  error_reporting(-1);
  ini_set('display_errors', 'On');

  $post = json_decode(file_get_contents('php://input'), true);
  $text = escapeshellarg($post['text']);
  chdir('../../');
  $cmd = "env/bin/python3.7 backend/src/predict.py $text 2>&1";
  $json = shell_exec($cmd);
  echo $json;
?>