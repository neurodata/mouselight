mkdir -p ~/.cloudvolume
mkdir -p ~/.cloudvolume/secrets

cat > ~/.cloudvolume/secrets/aws-secret.json << EOL
{
    "AWS_ACCESS_KEY_ID": "${AWS_KEY}",
    "AWS_SECRET_ACCESS_KEY": "${AWS_SEC_KEY}"
}
EOL

