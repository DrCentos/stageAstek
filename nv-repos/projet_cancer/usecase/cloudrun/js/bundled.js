import { initializeApp } from 'firebase/app';
import {
  onAuthStateChanged,
  signInWithEmailAndPassword,
  getAuth
} from 'firebase/auth';

const firebaseConfig = {
  apiKey: "AIzaSyCnibgCOBV6juAlMxec0ClPvsyXoFoszlM",
  authDomain: "glossy-precinct-371813.firebaseapp.com"
};
const app = initializeApp(firebaseConfig);
const auth = getAuth(app, {/* extra options */ });

document.addEventListener("DOMContentLoaded", () => {
  onAuthStateChanged(auth, (user) => {
      if (user) {
          document.getElementById("message").innerHTML = "Welcome, " + user.email;
      }
      else {
          document.getElementById("message").innerHTML = "No user signed in.";
      }
  });
  signIn();
});

function signIn() {
  const email = "nvallot@groupeastek.fr";
  const password = "nicolas";
  signInWithEmailAndPassword(auth, email, password)
      .catch((error) => {
          document.getElementById("message").innerHTML = error.message;
      });
}