const express = require('express');
const app = express();
const port = process.env.PORT || 4000; // Render should provide the PORT

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.listen(port, () => {
  console.log(`App listening on port ${port}`);
});
